#ifndef MNIST_H
#define MNIST_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <stdio.h>  

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <random>
#include <sstream>

#include <cublas_v2.h>
#include <cudnn.h>

// Definition and helper utilities

// Block width for CUDA kernels
#define BW 128
#define width 28
#define heigth 28
#define nTraining 60000
#define nTested 10000

#define gpu 0
#define iterations 10000
//#define random_seed -1
#define classify -1

int batch_size = 64;
bool pretrained = false;

#define learning_rate 0.001

using namespace std;


__global__
void FillVect(float *vec, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	vec[idx] = 1.0;
}

static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
	return (nominator + denominator - 1) / denominator;
}

/*Calcula los resultados de retropropagación de la pérdida de Softmax para cada resultado en un lote.
Utiliza los valores de softmax obtenidos del forward para calcular la diferencia.*/

__global__
void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;

	const int label_value = (int)(label[idx]);

	diff[idx * num_labels + label_value] -= 1.0;
}

//Capa de Convolucion
struct ConvoluLayer
{
	int in_channels, out_channels, kernel_size;
	int in_width, in_height, out_width, out_height;

	std::vector<float> vConv, vBias;

	ConvoluLayer(int in_channels_, int out_channels_, int kernel_size_,
		int in_w_, int in_h_)
	{
		vConv.resize(in_channels_ * kernel_size_ * kernel_size_ * out_channels_);
		vBias.resize(out_channels_);
		in_channels = in_channels_;
		out_channels = out_channels_;
		kernel_size = kernel_size_;
		in_width = in_w_;
		in_height = in_h_;
		out_width = in_w_ - kernel_size_ + 1;
		out_height = in_h_ - kernel_size_ + 1;
	}
};

//Capa Pooling
struct MaxPoolLayer
{
	int size;
	int stride;
	MaxPoolLayer(int size_, int stride_) 
	{
	  size = size_;
	  stride = stride_;
	}
};


//Capa full connected
struct FullConnectedLayer
{
	int inputs, outputs;
	vector<float> pneurons, pbias;

	FullConnectedLayer(int inputs_, int outputs_)
	{
		inputs = inputs_;
		outputs = outputs_;
		pneurons.resize(inputs * outputs);
		pbias.resize(outputs);
	}
};


struct Training
{
	cudnnHandle_t cudnnHandle; //es el puntero que alamneca a la libreria CUDNN
	cublasHandle_t cublasHandle; //es el puntero que alamneca a la libreria CUBLAS
	
	//descriptores
	cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor,
	conv2Tensor, conv2BiasTensor, pool2Tensor, fc1Tensor, fc2Tensor;
	//descriptores de filtro
	cudnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
	//descriptores de convoluciones
	cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
	//descriptores de operacion de convolucion
	cudnnConvolutionFwdAlgo_t conv1algo, conv2algo; //ejecutar la operación de convolución directa.
	//descriptores de filtro para Algo
	cudnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
	//datos
	cudnnConvolutionBwdDataAlgo_t conv2bwdalgo;
	//descriptores de pooling
	cudnnPoolingDescriptor_t poolDesc;
	//Activación
	cudnnActivationDescriptor_t fc1Activation;
	
	//redes full conectadas
	FullConnectedLayer& FullConnected_1;
	FullConnectedLayer& FullConnected_2;
	
	int m_gpuid;
	int m_batchSize;
	size_t m_workspaceSize;


	Training(int gpuid, int batch_size, ConvoluLayer& conv1, MaxPoolLayer& pool1, ConvoluLayer& conv2, MaxPoolLayer& pool2,FullConnectedLayer&  f1, FullConnectedLayer& f2):FullConnected_1(f1),FullConnected_2(f2)
	{
		m_gpuid = gpuid;
		m_batchSize = batch_size;


		cudaSetDevice(m_gpuid);
		cublasCreate(&cublasHandle);
		cudnnCreate(&cudnnHandle);

		cudnnCreateTensorDescriptor(&dataTensor);
		cudnnCreateTensorDescriptor(&conv1Tensor);
		cudnnCreateTensorDescriptor(&conv1BiasTensor);
		cudnnCreateTensorDescriptor(&pool1Tensor);
		cudnnCreateTensorDescriptor(&conv2Tensor);
		cudnnCreateTensorDescriptor(&conv2BiasTensor);
		cudnnCreateTensorDescriptor(&pool2Tensor);
		cudnnCreateTensorDescriptor(&fc1Tensor);
		cudnnCreateTensorDescriptor(&fc2Tensor);

		cudnnCreateActivationDescriptor(&fc1Activation);

		cudnnCreateFilterDescriptor(&conv1filterDesc);
		cudnnCreateFilterDescriptor(&conv2filterDesc);

		cudnnCreateConvolutionDescriptor(&conv1Desc);
		cudnnCreateConvolutionDescriptor(&conv2Desc);

		cudnnCreatePoolingDescriptor(&poolDesc);	

		//                         descriptor       formato            tipo de dato  #img       canal            w  h
		cudnnSetTensor4dDescriptor(conv1BiasTensor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1, conv1.out_channels,1, 1);
		cudnnSetTensor4dDescriptor(conv2BiasTensor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1, conv2.out_channels,1, 1);

		//                        descrriptor,modo, ,w,h,pHoriz,pVerti,horStride,verStride
		cudnnSetPooling2dDescriptor(poolDesc,CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,CUDNN_PROPAGATE_NAN,pool1.size, pool1.size,0, 0, pool1.stride, pool1.stride);

		//                         descriptor       formato            tipo de dato  #img       canal         w  h
		cudnnSetTensor4dDescriptor(pool2Tensor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batch_size, conv2.out_channels,			conv2.out_height / pool2.stride,conv2.out_width / pool2.stride);

		cudnnSetTensor4dDescriptor(fc1Tensor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batch_size, f1.outputs, 1, 1);

		cudnnSetTensor4dDescriptor(fc2Tensor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batch_size, f2.outputs, 1, 1);

		cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_CLIPPED_RELU,CUDNN_PROPAGATE_NAN, 0.0);

		//Devuelve la Cantidad de Espacio de Trabajo de memoria GPU Que el usuario NECESITA

		size_t workspace = 0;
		workspace = max(workspace, getAmountGPU_Forward(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
		workspace = max(workspace, getAmountGPU_Backward(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));

		workspace = max(workspace, getAmountGPU_Forward(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
		workspace = max(workspace, getAmountGPU_Backward(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));
	
		//cout<<workspace<<endl;
		m_workspaceSize = workspace;

	}

	~Training()
	{
		cudaSetDevice(m_gpuid);

		cublasDestroy(cublasHandle);
		cudnnDestroy(cudnnHandle);
		cudnnDestroyTensorDescriptor(dataTensor);
		cudnnDestroyTensorDescriptor(conv1Tensor);
		cudnnDestroyTensorDescriptor(conv1BiasTensor);
		cudnnDestroyTensorDescriptor(pool1Tensor);
		cudnnDestroyTensorDescriptor(conv2Tensor);
		cudnnDestroyTensorDescriptor(conv2BiasTensor);
		cudnnDestroyTensorDescriptor(pool2Tensor);
		cudnnDestroyTensorDescriptor(fc1Tensor);
		cudnnDestroyTensorDescriptor(fc2Tensor);
		cudnnDestroyActivationDescriptor(fc1Activation);
		cudnnDestroyFilterDescriptor(conv1filterDesc);
		cudnnDestroyFilterDescriptor(conv2filterDesc);
		cudnnDestroyConvolutionDescriptor(conv1Desc);
		cudnnDestroyConvolutionDescriptor(conv2Desc);
		cudnnDestroyPoolingDescriptor(poolDesc);
	}


	size_t getAmountGPU_Forward(ConvoluLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
		cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
		cudnnConvolutionFwdAlgo_t& algo)
	{
		size_t sizeInBytes = 0;

		int n = m_batchSize;
		int c = conv.in_channels;
		int h = conv.in_height;
		int w = conv.in_width;

		cudnnSetTensor4dDescriptor(srcTensorDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,n, c, h, w);

		cudnnSetFilter4dDescriptor(filterDesc,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,conv.out_channels,conv.in_channels,conv.kernel_size,
			conv.kernel_size);

		cudnnSetConvolution2dDescriptor(convDesc,0, 0,1, 1,1, 1,CUDNN_CONVOLUTION);// función inicializa un objeto descriptor de convolución previamente creado en una correlación 2D
		
		// Find dimension of convolution output
		cudnnGetConvolution2dForwardOutputDim(convDesc,srcTensorDesc,filterDesc,&n, &c, &h, &w);//devuelve las dimesiones 4d de un 2d

		cudnnSetTensor4dDescriptor(dstTensorDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,n, c,h, w);
		//devuelve la mejor heuristica para obtener el algoritmo mas adecuado
		cudnnGetConvolutionForwardAlgorithm(cudnnHandle,srcTensorDesc,filterDesc,convDesc,dstTensorDesc,CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			0,&algo);
		//se utiliza para obtener la cantidad de memoria necesaria para el algoritmo
		cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,srcTensorDesc,filterDesc,convDesc,dstTensorDesc,	algo,
			&sizeInBytes);

		cout<<"size: "<<sizeInBytes<<endl;
		return sizeInBytes;
	}

	size_t getAmountGPU_Backward(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
		cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo)
	{
		size_t sizeInBytes = 0, tmpsize = 0;

		if (falgo)
		{
			cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo);
			cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,*falgo, &tmpsize);
			sizeInBytes = max(sizeInBytes, tmpsize);
		}

		if (dalgo)
		{
			cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo);
			cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,*dalgo, &tmpsize);
			sizeInBytes = max(sizeInBytes, tmpsize);
		}
		return sizeInBytes;
	}

	void Forward(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
		float *fc2, float *result,float *pconv1, float *pconv1bias,float *pconv2, float *pconv2bias,float *pfc1, float *pfc1bias,
		float *pfc2, float *pfc2bias, void *workspace, float *onevec)
	{
		float alpha = 1.0, beta = 0.0; //factores de escala para ejecutar la convolucion
		cudaSetDevice(m_gpuid);

		// Foward para la primera capa de convolución
		cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,data, conv1filterDesc, pconv1, conv1Desc,
			conv1algo, workspace, m_workspaceSize, &beta,conv1Tensor, conv1);

		cudnnAddTensor(cudnnHandle, &alpha, conv1BiasTensor,pconv1bias, &alpha, conv1Tensor, conv1);

		//Primer pooling

		cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor,conv1, &beta, pool1Tensor, pool1);

		// Foward para la segunda capa de convolución
		cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor,pool1, conv2filterDesc, pconv2, conv2Desc,conv2algo, workspace, m_workspaceSize, &beta,conv2Tensor, conv2);

		cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor,pconv2bias, &alpha, conv2Tensor, conv2);

		// Segundo Pooling
		cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor,conv2, &beta, pool2Tensor, pool2);

		//forward para (fc1 = pfc1(T)*pool2) 
		cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,FullConnected_1.outputs, m_batchSize, FullConnected_1.inputs,&alpha,pfc1, FullConnected_1.inputs,pool2, FullConnected_1.inputs,
			&beta,fc1, FullConnected_1.outputs);
		
		cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,FullConnected_1.outputs, m_batchSize, 1,&alpha,pfc1bias, FullConnected_1.outputs,onevec, 1,&alpha,
			fc1, FullConnected_1.outputs);

	    cudnnActivationForward(cudnnHandle, fc1Activation, &alpha,fc1Tensor, fc1, &beta, fc1Tensor, fc1relu);

	    //forward para (fc2 = pfc2'*fc1relu)
	    cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,FullConnected_2.outputs, m_batchSize, FullConnected_2.inputs,&alpha,	pfc2, FullConnected_2.inputs,fc1relu, FullConnected_2.inputs,
			&beta,fc2, FullConnected_2.outputs);

	    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,FullConnected_2.outputs, m_batchSize, 1,&alpha,pfc2bias, FullConnected_2.outputs,onevec, 1,&alpha,fc2, FullConnected_2.outputs);

	    //Softmax no es una función de pérdida, sino que se utiliza para hacer que la salida de una red neural sea más "compatible".
		cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,&alpha, fc2Tensor, fc2, &beta, fc2Tensor, result);
	}

	void Backpropagation(ConvoluLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvoluLayer& layer_conv2, MaxPoolLayer& layer_pool2,
		float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,float *fc2, float *fc2smax, float *dloss_data,float *pconv1, float *pconv1bias,float *pconv2, float *pconv2bias,float *pfc1, float *pfc1bias,
		float *pfc2, float *pfc2bias,float *gconv1, float *gconv1bias, float *dpool1,float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
		float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,float *gfc2, float *gfc2bias, float *dfc2,void *workspace, float *onevec)
	{
		float alpha = 1.0;
		float beta = 0.0;
		float scalVal = 1.0 /(1.0*m_batchSize);
		//cout<<alpha<<beta<<scalVal<<endl;	
		cudaSetDevice(m_gpuid);

		cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float)* m_batchSize * FullConnected_2.outputs, cudaMemcpyDeviceToDevice);

		SoftmaxLossBackprop << < RoundUp(m_batchSize, BW), BW >> >(labels, FullConnected_2.outputs, m_batchSize, dloss_data);

		//hallando la gradiente
		cublasSscal(cublasHandle, FullConnected_2.outputs * m_batchSize, &scalVal, dloss_data, 1);

		//gfc2 = (fc1relu * dfc2smax')
		cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, FullConnected_2.inputs, FullConnected_2.outputs, m_batchSize,&alpha, fc1relu, FullConnected_2.inputs, dloss_data, FullConnected_2.outputs, &beta, gfc2, FullConnected_2.inputs);

		//gfc2bias = dfc2smax * 1_vec
		cublasSgemv(cublasHandle, CUBLAS_OP_N, FullConnected_2.outputs, m_batchSize,&alpha, dloss_data, FullConnected_2.outputs, onevec, 1, &beta, gfc2bias, 1);
 		
 		//pfc2*dfc2smax (500x10*10xN)
		cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, FullConnected_2.inputs, m_batchSize, FullConnected_2.outputs,&alpha, pfc2, FullConnected_2.inputs, dloss_data, FullConnected_2.outputs, &beta, dfc2, FullConnected_2.inputs);

		//Ahora activamos cudnn para realizar el backpropagation
		cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha,fc1Tensor, fc1relu, fc1Tensor, dfc2,fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu);
		//gfc1 = (pool2 * dfc1relu')
		cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, FullConnected_1.inputs, FullConnected_1.outputs, m_batchSize,&alpha, pool2, FullConnected_1.inputs, dfc1relu, FullConnected_1.outputs, &beta, gfc1, FullConnected_1.inputs);
		//gfc1bias = dfc1relu * 1_vec
		cublasSgemv(cublasHandle, CUBLAS_OP_N, FullConnected_1.outputs, m_batchSize,&alpha, dfc1relu, FullConnected_1.outputs, onevec, 1, &beta, gfc1bias, 1);
		// pfc1*dfc1relu (800x500*500xN)
		cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, FullConnected_1.inputs, m_batchSize, FullConnected_1.outputs,&alpha, pfc1, FullConnected_1.inputs, dfc1relu, FullConnected_1.outputs, &beta, dfc1, FullConnected_1.inputs);
		//saca las gradientes para la capa pooling 2
		cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,pool2Tensor, pool2, pool2Tensor, dfc1,conv2Tensor, conv2, &beta, conv2Tensor, dpool2);		
		// función calcula el gradiente de la función de convolución con respecto al bias
		cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor,dpool2, &beta, conv2BiasTensor, gconv2bias);
		//función calcula el gradiente de convolución con respecto a los coeficientes de filtro usando el algoritmo especificado,
		cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1Tensor,pool1, conv2Tensor, dpool2, conv2Desc,conv2bwfalgo, workspace, m_workspaceSize,&beta, conv2filterDesc, gconv2);
		//función calcula el gradiente de convolución para la ultima salida de la segunda capa de convolucion
		cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2filterDesc,pconv2, conv2Tensor, dpool2, conv2Desc,conv2bwdalgo, workspace, m_workspaceSize,&beta, pool1Tensor, dconv2);

		//saca las gradientes para la capa pooling 1
		cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,pool1Tensor, pool1, pool1Tensor, dconv2,conv1Tensor, conv1, &beta, conv1Tensor, dpool1);		
		//Backpropagation para la primera capa de convolución - no se necesita realizar  cudnnConvolutionBackwardData porque no pasaremos los valores a otra capa
		cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,dpool1, &beta, conv1BiasTensor, gconv1bias);
		cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,	data, conv1Tensor, dpool1, conv1Desc,conv1bwfalgo, workspace, m_workspaceSize,&beta, conv1filterDesc, gconv1);
	}

	void UpdateWeigth(float _learning_rate, ConvoluLayer& conv1, ConvoluLayer& conv2, float *pconv1, float *pconv1bias, float *pconv2, float *pconv2bias,float *pfc1, float *pfc1bias, float *pfc2, float *pfc2bias, float *gconv1, float *gconv1bias, float *gconv2, float *gconv2bias, float *gfc1, float *gfc1bias, float *gfc2, float *gfc2bias)
	{
		float alpha = -learning_rate;
		cudaSetDevice(m_gpuid);
		//actualizamos los miembros datos con los valores obtenidos en el BackPropagation
		cublasSaxpy(cublasHandle, (int)conv1.vConv.size(),&alpha, gconv1, 1, pconv1, 1);
		cublasSaxpy(cublasHandle, (int)conv1.vBias.size(),&alpha, gconv1bias, 1, pconv1bias, 1);

		cublasSaxpy(cublasHandle, (int)conv2.vConv.size(),&alpha, gconv2, 1, pconv2, 1);
		cublasSaxpy(cublasHandle, (int)conv2.vBias.size(),&alpha, gconv2bias, 1, pconv2bias, 1);

		cublasSaxpy(cublasHandle, (int)FullConnected_1.pneurons.size(),&alpha, gfc1, 1, pfc1, 1);
		cublasSaxpy(cublasHandle, (int)FullConnected_1.pbias.size(),&alpha, gfc1bias, 1, pfc1bias, 1);

		cublasSaxpy(cublasHandle, (int)FullConnected_2.pneurons.size(),&alpha, gfc2, 1, pfc2, 1);
		cublasSaxpy(cublasHandle, (int)FullConnected_2.pbias.size(),&alpha, gfc2bias, 1, pfc2bias, 1);
	}
};
#endif