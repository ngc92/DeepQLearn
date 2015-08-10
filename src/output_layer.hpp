#pragma once

#include "layer.hpp"

// class for the input layer
template<class T>
class OutputLayer : public ILayer<T>
{
public:
	using typename ILayer<T>::WP_ILayer;
	using typename ILayer<T>::range_t;
	using typename ILayer<T>::const_range_t;
	
	
	OutputLayer( unsigned neurons );
	
	// propagate signal through the layer
	void forward() override;
	void backward() override;
	
	// build up connections
	void setPreviousLayer( const WP_ILayer& prev ) override;
	void setNextLayer( const WP_ILayer& next ) override {assert(0);};
	
	// infos 
	unsigned getNumNeurons() const override { return mNumNeurons; };
	unsigned getNumInputs()  const override { return mNumNeurons; };
	
	const WP_ILayer& getPreviousLayer() const override { return mPreviousLayer; };
	const WP_ILayer& getNextLayer()     const override { return EmptyNext; };
	
private:
	
	// get access to layer data
	range_t getOutputMutable()   override { return range_t{mNeuronOut.data(),   mNeuronOut.data()   + mNeuronOut.size()};   };
	range_t getWeightsMutable()  override { return range_t{}; };
	range_t getBiasMutable()     override { return range_t{}; };
	range_t getGradientMutable()    override { return range_t{}; };
	
	
	static const WP_ILayer EmptyNext;
	
	unsigned mNumNeurons;
	
	// neurons
	std::vector<T> mNeuronOut;
	std::vector<T> mGradient;
	
	// connected layers
	WP_ILayer mPreviousLayer;
};

template<class T>
OutputLayer<T>::OutputLayer( unsigned neurons ) : 
	mNumNeurons( neurons ), 
	// allocate memory and zero initialize
	mNeuronOut( neurons ),
	mGradient( neurons )
{
	
}

template<class T>
void OutputLayer<T>::setPreviousLayer( const WP_ILayer& prev )
{
	// for now, do not allow to change the prev layer later on
	assert( mPreviousLayer.expired() );
	
	// check layer compatibility
	assert( prev.lock()->getNumInputs() == getNumNeurons() );
	
	mPreviousLayer = prev;
}

template<class T>
void OutputLayer<T>::forward()
{
}

template<class T>
void OutputLayer<T>::backward()
{
}

template<typename T>
const typename ILayer<T>::WP_ILayer OutputLayer<T>::EmptyNext{};

