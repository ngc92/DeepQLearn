#ifndef NETWORK_HPP_INCLUDED
#define NETWORK_HPP_INCLUDED

#include <memory>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <random>

#include "layer.hpp"
#include "layer_factory.hpp"

// fann function that might be usefully copied?

struct LayerInfo
{
	LayerInfo() = default;
	LayerInfo(const std::string& t, int n ) : type(t), neurons(n)
	{
	}
	
	std::string type;
	int neurons;
};

template<class T>
class Network
{
	typedef std::shared_ptr<ILayer<T>> P_Layer;
	
public:
	Network( std::vector<LayerInfo> layers );
	Network( Network&& ) = default;
	// made uncopyable (shallow copy). For a deep copy, use clone
	Network& operator=(const Network<T>& o) = delete;
	
	Network<T> clone() const;
	
	void forward( const std::vector<T>& input );
	void backward( const std::vector<T>& gradient );
	
	boost::iterator_range<const T*> getOutput() const { return mLayers.back()->getOutput(); };
	
	// info functions
	unsigned getNumInputs() const { return mNumInputs; };
	unsigned getNumOutputs() const { return mNumOutputs; };
	unsigned getNumLayers() const { return mLayers.size(); };
	
	const P_Layer& getLayer( unsigned id ) { return mLayers.at(id); };
	
	// functions to come:
	/*	save / load
		getter for all weights, gradients etc
	*/
	
private:
	// this c'tor exists, because it can be used as basis for implementing the deep copy
	Network( const Network<T>& other ) = default;
	
	unsigned mNumInputs;
	unsigned mNumOutputs;
	
	// configurations
	bool mSaveOutput = true;
	
	// the layers
	std::vector<P_Layer> mLayers;
	
};


// most simple network setup
template<class T>
Network<T>::Network(std::vector<LayerInfo> layers)
{
	// if last layer is not an output layer, add 
	if( layers.back().type != "output" )
	{
		layers.emplace_back("output", layers.at(layers.size()-1).neurons);
	}
	assert( layers.size() >= 2 );
	// check that first is input (we made sure last was output)
	assert( layers.front().type == "input" );
	
	mLayers.push_back(createLayer<T>(0, layers.front().neurons, layers.front().type));
	
	// generate hidden layers
	for(unsigned i = 1; i < layers.size(); ++i)
	{
		auto layer = createLayer<T>(layers.at(i-1).neurons, layers.at(i).neurons, layers.at(i).type);
		mLayers.push_back(layer);
	}
	
	mNumInputs = mLayers.front()->getNumNeurons();
	mNumOutputs = mLayers.back()->getNumNeurons();
	
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
}


template<class T>
void Network<T>::backward( const std::vector<T>& gradient )
{
	for(auto& l : mLayers )
		l->resetGradient();
	
	mLayers.back()->setGradient(gradient);
	
	for(auto it = mLayers.rbegin(); it != mLayers.rend(); ++it)
	{
		(*it)->backward();
	}
}

template<class T>
Network<T> Network<T>::clone() const
{
	Network<T> newnet {*this};
	for(auto& l : newnet.mLayers)
	{
		l = l->clone();
	}
	
	// update prev/next pointers
	for(unsigned i = 1; i < mLayers.size(); ++i)
	{
		newnet.mLayers.at(i)->setPreviousLayer( newnet.mLayers.at(i-1) );
		newnet.mLayers.at(i-1)->setNextLayer( newnet.mLayers.at(i) );
	}
	
	return newnet;
}

#endif // NETWORK_HPP_INCLUDED
