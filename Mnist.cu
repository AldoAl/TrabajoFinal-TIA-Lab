
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "ReadFile.h"
#include "Mnist.cuh"

using namespace std;



int main()
{
	string train_images = "data/train-images.idx3-ubyte";
	string train_labels = "data/train-labels.idx1-ubyte";
	string test_images = "data/t10k-images.idx3-ubyte";
	string test_labels = "data/t10k-labels.idx1-ubyte";
	
	size_t height;
	size_t v_width;
	size_t channels=1;
	int batch_size=64;
	float learningRate=0.001;
	int random_seed =-1;
	int nIteration=10000;
	//parte de forward para convolucional
	float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
	//parte capas full conectadas
	float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
	float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;
	//parte de backward 
	float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
	float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
	//Diferenciales
	float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;

	float *d_onevec;
	void *d_cudnn_workspace = nullptr;

	printf("Aprendizaje...\n");

	size_t train_size = ReadUByteDataset(train_images.c_str(),train_labels.c_str(),nullptr,nullptr,v_width,height);
	size_t test_size = ReadUByteDataset(test_images.c_str(),test_labels.c_str(),nullptr,nullptr, v_width, height);

	cout<<train_size<<endl;
	cout<<test_size<<endl;

	size_t sizeTrainImages = train_size * v_width * height * channels;
	size_t sizeTestImages = test_size * v_width * height * channels;
	vector<int> v_train_images(sizeTrainImages), v_train_labels(train_size);
	vector<int> v_test_images(sizeTrainImages), v_test_labels(test_size);
	
	cout<<"tam de imagenes de entrenamiento: "<<sizeTrainImages<<endl;
	cout<<"tam de imagenes de testeo: "<<sizeTestImages<<endl;
	cout<<"batch_size: "<<batch_size<<" Iteraciones: "<<nIteration<<endl;

	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	cout<<"gpus: "<<num_gpus<<endl;
   

	ConvoluLayer conv1((int)channels, 20, 5, (int)v_width, (int)height);
	
	MaxPoolLayer pool1(2, 2);
	
	ConvoluLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);

	MaxPoolLayer pool2(2, 2);
	FullConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride),
		500);
	FullConnectedLayer fc2(fc1.outputs, 10);
	//objeto de entrenamiento
	Training  TrainiMnist(num_gpus, batch_size, conv1, pool1, conv2, pool2, fc1, fc2);
	
	//Creamos valores aleatorios
	random_device rd;
	mt19937 ale(random_seed < 0 ? rd() : (int)random_seed);
	float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
	uniform_real_distribution<> dconv1(-wconv1, wconv1);
	float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
	uniform_real_distribution<> dconv2(-wconv2, wconv2);
	float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
	uniform_real_distribution<> dfc1(-wfc1, wfc1);
	float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
	uniform_real_distribution<> dfc2(-wfc2, wfc2);
	
	for (auto iter : conv1.vConv)
		iter = (float)dconv1(ale);
	for (auto iter : conv1.vBias)
		iter = (float)dconv1(ale);
	for (auto iter : conv2.vConv)
		iter = (float)dconv2(ale);
	for (auto iter : conv2.vBias)
		iter = (float)dconv2(ale);
	for (auto iter : fc1.pneurons)
		iter = (float)dfc1(ale);
	for (auto iter : fc1.pbias)
		iter = (float)dfc1(ale);
	for (auto iter : fc2.pneurons)
		iter = (float)dfc2(ale);
	for (auto iter : fc2.pbias)
		iter = (float)dfc2(ale);

	cudaMalloc(&d_data, sizeof(float)* TrainiMnist.m_batchSize * channels * height * width);
	cudaMalloc(&d_labels, sizeof(float)* TrainiMnist.m_batchSize * 1 * 1 * 1);
	cudaMalloc(&d_conv1, sizeof(float)* TrainiMnist.m_batchSize * conv1.out_channels * conv1.out_height * conv1.out_width);
	cudaMalloc(&d_pool1, sizeof(float)* TrainiMnist.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride));
	cudaMalloc(&d_conv2, sizeof(float)* TrainiMnist.m_batchSize * conv2.out_channels * conv2.out_height * conv2.out_width);
	cudaMalloc(&d_pool2, sizeof(float)* TrainiMnist.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride));
	cudaMalloc(&d_fc1, sizeof(float)* TrainiMnist.m_batchSize * fc1.outputs);
	cudaMalloc(&d_fc1relu, sizeof(float)* TrainiMnist.m_batchSize * fc1.outputs);
	cudaMalloc(&d_fc2, sizeof(float)* TrainiMnist.m_batchSize * fc2.outputs);
	cudaMalloc(&d_fc2smax, sizeof(float)* TrainiMnist.m_batchSize * fc2.outputs);

	cudaMalloc(&d_pconv1, sizeof(float)* conv1.vConv.size());
	cudaMalloc(&d_pconv1bias, sizeof(float)* conv1.vBias.size());
	cudaMalloc(&d_pconv2, sizeof(float)* conv2.vConv.size());
	cudaMalloc(&d_pconv2bias, sizeof(float)* conv2.vBias.size());
	cudaMalloc(&d_pfc1, sizeof(float)* fc1.pneurons.size());
	cudaMalloc(&d_pfc1bias, sizeof(float)* fc1.pbias.size());
	cudaMalloc(&d_pfc2, sizeof(float)* fc2.pneurons.size());
	cudaMalloc(&d_pfc2bias, sizeof(float)* fc2.pbias.size());

	cudaMalloc(&d_gconv1, sizeof(float)* conv1.vConv.size());
	cudaMalloc(&d_gconv1bias, sizeof(float)* conv1.vBias.size());
	cudaMalloc(&d_gconv2, sizeof(float)* conv2.vConv.size());
	cudaMalloc(&d_gconv2bias, sizeof(float)* conv2.vBias.size());
	cudaMalloc(&d_gfc1, sizeof(float)* fc1.pneurons.size());
	cudaMalloc(&d_gfc1bias, sizeof(float)* fc1.pbias.size());
	cudaMalloc(&d_gfc2, sizeof(float)* fc2.pneurons.size());
	cudaMalloc(&d_gfc2bias, sizeof(float)* fc2.pbias.size());

	cudaMalloc(&d_dpool1, sizeof(float)* TrainiMnist.m_batchSize * conv1.out_channels * conv1.out_height * conv1.out_width);
	cudaMalloc(&d_dpool2, sizeof(float)* TrainiMnist.m_batchSize * conv2.out_channels * conv2.out_height * conv2.out_width);
	cudaMalloc(&d_dconv2, sizeof(float)* TrainiMnist.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride));
	cudaMalloc(&d_dfc1, sizeof(float)* TrainiMnist.m_batchSize * fc1.inputs);
	cudaMalloc(&d_dfc1relu, sizeof(float)* TrainiMnist.m_batchSize * fc1.outputs);
	cudaMalloc(&d_dfc2, sizeof(float)* TrainiMnist.m_batchSize * fc2.inputs);
	cudaMalloc(&d_dfc2smax, sizeof(float)* TrainiMnist.m_batchSize * fc2.outputs);
	cudaMalloc(&d_dlossdata, sizeof(float)* TrainiMnist.m_batchSize * fc2.outputs);

	cudaMalloc(&d_onevec, sizeof(float)* TrainiMnist.m_batchSize);

	if (TrainiMnist.m_workspaceSize > 0)
		cudaMalloc(&d_cudnn_workspace, TrainiMnist.m_workspaceSize);

	cudaMemcpyAsync(d_pconv1, &conv1.vConv[0], sizeof(float)* conv1.vConv.size(), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pconv1bias, &conv1.vBias[0], sizeof(float)* conv1.vBias.size(), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pconv2, &conv2.vConv[0], sizeof(float)* conv2.vConv.size(), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pconv2bias, &conv2.vBias[0], sizeof(float)* conv2.vBias.size(), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0], sizeof(float)* fc1.pneurons.size(), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0], sizeof(float)* fc1.pbias.size(), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0], sizeof(float)* fc2.pneurons.size(), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0], sizeof(float)* fc2.pbias.size(), cudaMemcpyHostToDevice);
	
	FillVect << <RoundUp(TrainiMnist.m_batchSize, BW), BW >> >(d_onevec, TrainiMnist.m_batchSize);

	vector<float> train_images_float(v_train_images.size());
	vector<float> train_labels_float(train_size);

	for (int i = 0; i < train_size*channels*v_width*height; ++i)
	{
		train_images_float[i] = (1.0*train_images[i])/255.0f;
	}

	for (size_t i = 0; i < train_size; ++i)
	{
		train_labels_float[i] = 1.0*train_labels[i];
	}

	cudaDeviceSynchronize();
	
	auto t1 = chrono::high_resolution_clock::now();
	
	for (int i = 0; i < nIteration; ++i)
	{
		int imageid = i % (train_size / TrainiMnist.m_batchSize);
		cudaMemcpyAsync(d_data, &train_images_float[imageid * TrainiMnist.m_batchSize * width*height*channels],sizeof(float)* TrainiMnist.m_batchSize * channels * width * height, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_labels, &train_labels_float[imageid * TrainiMnist.m_batchSize],sizeof(float)* TrainiMnist.m_batchSize, cudaMemcpyHostToDevice);
		
		TrainiMnist.Forward(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,d_cudnn_workspace, d_onevec);

		TrainiMnist.Backpropagation(conv1, pool1, conv2, pool2,d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
			d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_gfc1, d_gfc1bias,d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);

		TrainiMnist.UpdateWeigth(learningRate, conv1, conv2,d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias);
	}
	cudaDeviceSynchronize();
	auto t2 = chrono::high_resolution_clock::now();
	cout<<"Tiempo de demora: "<<chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000.0f<<"ms"<<endl;


	float classification_error = 1.0;
	int classifications = -1;

	if (classifications < 0)
	{
		classifications = (int)test_size;
	}
	//Testeamos la red neuronal con los datos de prueba
	if (classifications > 0)
	{
		Training TestMnist(num_gpus, 1, conv1, pool1, conv2, pool2, fc1, fc2);

		int num_errors = 0;
		for (int i = 0; i < classifications; ++i)
		{
			vector<float> data(width * height);// 28*28 =784 

			//Normalizamos los valores a 1 y a 0
			for (int j = 0; j < width * height; ++j)
			{
				data[j] = (1.0*v_test_images[i * width*height*channels + j]) / 255.0f;
			}
			cudaMemcpyAsync(d_data, &data[0], sizeof(float)* width * height, cudaMemcpyHostToDevice);

			TestMnist.Forward(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias,d_pfc2, d_pfc2bias, d_cudnn_workspace, d_onevec);
			
			vector<float> Desired_val(10);
			cudaMemcpy(&Desired_val[0], d_fc2smax, sizeof(float)* 10, cudaMemcpyDeviceToHost);

			int chosen = 0;
			for (int id = 1; id < 10; ++id)
			{
				if (Desired_val[chosen] < Desired_val[id]) chosen = id;
			}

			if (chosen != test_labels[i])
			{
				++num_errors;
			}
		}
		classification_error = (float)num_errors / (float)classifications;
		double precision = classifications - classification_error;

		cout<<"Error en el Testeo de :"<<classification_error*100.0<<"%  en "<<classifications<<" iteraciones."<<endl;
	}
	cudaFree(d_data);
	cudaFree(d_conv1);
	cudaFree(d_pool1);
	cudaFree(d_conv2);
	cudaFree(d_pool2);
	cudaFree(d_fc1);
	cudaFree(d_fc2);
	cudaFree(d_pconv1);
	cudaFree(d_pconv1bias);
	cudaFree(d_pconv2);
	cudaFree(d_pconv2bias);
	cudaFree(d_pfc1);
	cudaFree(d_pfc1bias);
	cudaFree(d_pfc2);
	cudaFree(d_pfc2bias);
	cudaFree(d_gconv1);
	cudaFree(d_gconv1bias);
	cudaFree(d_gconv2);
	cudaFree(d_gconv2bias);
	cudaFree(d_gfc1);
	cudaFree(d_gfc1bias);
	cudaFree(d_dfc1);
	cudaFree(d_gfc2);
	cudaFree(d_gfc2bias);
	cudaFree(d_dfc2);
	cudaFree(d_dpool1);
	cudaFree(d_dconv2);
	cudaFree(d_dpool2);
	cudaFree(d_labels);
	cudaFree(d_dlossdata);
	cudaFree(d_onevec);
	cout<<"Fin del Programa."<<endl;
	return 0;

}
