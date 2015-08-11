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
	
	void forward( const std::vector<T>& input );
	void backward( const std::vector<T>& gradient );
	void setDesiredOutput( std::vector<T> out );
	
	const std::vector<T>& getOutput() const { return mLastOutput; };
	
	// info functions
	unsigned getNumInputs() const { return mNumInputs; };
	unsigned getNumOutputs() const { return mNumOutputs; };
	unsigned getNumLayers() const { return mLayers.size(); };
	
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
	
	if( mSaveOutput )
	{
		mLayers.back()->getOutput(mLastOutput);
	}
}


template<class T>
void Network<T>::backward( const std::vector<T>& gradient )
{
	mLayers.back()->setGradient(gradient);
	
	for(auto it = mLayers.rbegin(); it != mLayers.rend(); ++it)
	{
		(*it)->backward();
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
		
		
	}
	
	mMSE = sqerrsum / out.size();
}


#endif // NETWORK_HPP_INCLUDED
