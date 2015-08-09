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

// fann function that might be usefully copied?

template<class T>
class Network
{
	typedef std::shared_ptr<ILayer<T>> P_Layer;
	
public:
	Network( std::vector<int> layers );
	
	void run(  );
	void forward( const std::vector<T>& input );
	
	const std::vector<T>& getOutput() const { return mLastOutput; };
	
private:
	unsigned mNumInputs;
	unsigned mNumOutputs;
	
	// configurations
	bool mSaveOutput = true;
	
	void calculateError();
	
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
	
	#ifndef NDEBUG
	std::cout << "creating network with " << layers.size() << " layers\n";
	std::cout << "  input: " << mNumInputs << "\n";
	#endif
	
	// generate input layer
	mLayers.push_back(std::make_shared<InputLayer<T>>( mNumInputs ));
	
	// generate hidden layers
	for(unsigned i = 1; i < layers.size(); ++i)
	{
		auto layer = std::make_shared<FCLayer<T, activation::tanh>>(layers.at(i-1), layers.at(i));
		mLayers.push_back(layer);
		#ifndef NDEBUG
		std::cout << "  layer: " << layers.at(i) << "neurons, 1 bias" << "\n";
		#endif
	}
	
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


/*
// compute the error in the last layer
void fann_compute_MSE(struct fann *ann, fann_type * desired_output)
{

#ifdef DEBUGTRAIN
	printf("\ncalculate errors\n");
#endif
	error_it = error_begin + (last_layer_begin - first_neuron);

	for(; last_layer_begin != last_layer_end; last_layer_begin++)
	{
		neuron_value = last_layer_begin->value;
		neuron_diff = *desired_output - neuron_value;

		neuron_diff = fann_update_MSE(ann, last_layer_begin, neuron_diff);

		if(ann->train_error_function)
		{
			if(neuron_diff < -.9999999)
				neuron_diff = -17.0;
			else if(neuron_diff > .9999999)
				neuron_diff = 17.0;
			else
				neuron_diff = (fann_type) log((1.0 + neuron_diff) / (1.0 - neuron_diff));
		}

		*error_it = fann_activation_derived(last_layer_begin->activation_function,
											last_layer_begin->activation_steepness, neuron_value,
											last_layer_begin->sum) * neuron_diff;

		desired_output++;
		error_it++;

		ann->num_MSE++;
	}
}
*/


#endif // NETWORK_HPP_INCLUDED
