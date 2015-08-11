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
	void updateWeights( const T& eta ) override {};
	
	// build up connections
	void setPreviousLayer( const WP_ILayer& prev ) override;
	void setNextLayer( const WP_ILayer& next ) override {assert(0);};
	
	// infos 
	unsigned getNumNeurons() const override   { return mNumNeurons; };
	unsigned getNumInputs()  const override   { return mNumNeurons; };
	const char* getLayerType() const override { return "output";    };
	
	const WP_ILayer& getPreviousLayer() const override { return mPreviousLayer; };
	const WP_ILayer& getNextLayer()     const override { return EmptyNext; };
	
private:
	
	// get access to layer data
	range_t getOutputMutable()   override { return range_t{mNeuronOut.data(),   mNeuronOut.data()   + mNeuronOut.size()};   };
	range_t getWeightsMutable()  override { return range_t{}; };
	range_t getGradientMutable()    override { return range_t{mGradient.data(), mGradient.data() + mGradient.size()}; };
	
	
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
	assert( prev.lock()->getNumNeurons() == getNumNeurons() );
	
	mPreviousLayer = prev;
}

template<class T>
void OutputLayer<T>::forward()
{
	// propagate output unchanged!
	/// \todo this construction might be bad for performance!
	boost::copy( mPreviousLayer.lock()->getOutput(), mNeuronOut.begin() );
}

template<class T>
void OutputLayer<T>::backward()
{
	// output gets the gradient / error set externally. no work here
}

template<typename T>
const typename ILayer<T>::WP_ILayer OutputLayer<T>::EmptyNext{};

