#ifndef NETWORK_HPP_INCLUDED
#define NETWORK_HPP_INCLUDED

#include <memory>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <random>

#include "layer.hpp"
#include "fc_layer.hpp"
#include "input_layer.hpp"
#include "nl_layer.hpp"
#include "output_layer.hpp"

// fann function that might be usefully copied?

template<class T>
class Network
{
	typedef std::shared_ptr<ILayer<T>> P_Layer;
	
public:
	Network( std::vector<int> layers );
	
	void run(  );
	void forward( const std::vector<T>& input );
	void setDesiredOutput( std::vector<T> out );
	
	const std::vector<T>& getOutput() const { return mLastOutput; };
	
private:
	unsigned mNumInputs;
	unsigned mNumOutputs;
	
	// configurations
	bool mSaveOutput = true;
	
	// error tracking
	T mMSE = T(0);
	
	// the layers
	std::vector<P_Layer> mLayers;
	
	std::vector<T> mLastOutput;
	
};


// most simple network setup
template<class T>
Network<T>::Network(std::vector<int> layers)
{
	assert( layers.size() >= 2 );
	
	mNumInputs = layers.front();
	mNumOutputs = layers.back();
	
	// generate input layer
	mLayers.push_back(std::make_shared<InputLayer<T>>( mNumInputs ));
	
	// generate hidden layers
	for(unsigned i = 1; i < layers.size(); ++i)
	{
		auto layer = std::make_shared<FCLayer<T>>(layers.at(i-1), layers.at(i));
		mLayers.push_back(layer);
		
		auto nlayer = std::make_shared<NLLayer<T, activation::tanh>>(layers.at(i));
		mLayers.push_back(nlayer);
	}

	mLayers.push_back(std::make_shared<OutputLayer<T>>( mNumOutputs ));
	
	// connect the layers
	for(unsigned i = 1; i < layers.size(); ++i)
	{
		mLayers.at(i)->setPreviousLayer( mLayers.at(i-1) );
		mLayers.at(i-1)->setNextLayer( mLayers.at(i) );
	}
	
	// initialize the weights
	std::default_random_engine random;
	std::uniform_real_distribution<T> dst(-1, 1);
	
	for(auto& l : mLayers)
		l->randomizeWeights( std::bind(dst, random) );
}

template<class T>
void Network<T>::forward( const std::vector<T>& input )
{
	mLayers.front()->setOutput(input);
	for(auto& layer : mLayers )
		layer->forward();
	
	if( mSaveOutput )
	{
		mLayers.back()->getOutput(mLastOutput);
	}
}

template<class T>
void Network<T>::setDesiredOutput( std::vector<T> out )
{
	auto rout = mLayers.back().getOutput();
	assert( rout.size() == out.size() );
	
	T sqerrsum = 0;
	
	for( unsigned i = 0; i < out.size(); ++i)
	{
		out[i] -= rout[i];
		sqerrsum += out[i] * out[i];
		/*
		if(ann->train_error_function)
		{
			if(neuron_diff < -.9999999)
				neuron_diff = -17.0;
			else if(neuron_diff > .9999999)
				neuron_diff = 17.0;
			else
				neuron_diff = (fann_type) log((1.0 + neuron_diff) / (1.0 - neuron_diff));
		}*/
		
		/**error_it = fann_activation_derived(last_layer_begin->activation_function,
											last_layer_begin->activation_steepness, neuron_value,
											last_layer_begin->sum) * neuron_diff;
		*/
	}
	
	mMSE = sqerrsum / out.size();
}


#endif // NETWORK_HPP_INCLUDED
