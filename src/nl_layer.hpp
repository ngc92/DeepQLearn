#pragma once
#ifndef NL_LAYER_HPP_INCLUDED
#define NL_LAYER_HPP_INCLUDED

// Fully Connected Layer 
#include "layer.hpp"

namespace activation
{
	template<class T>
	class tanh
	{
	public:
		T operator()( const T& t) 
		{
			return std::tanh(t);
		}
		
		T deriv( const T& t )
		{
			T ch = std::cosh(t);
			return 1.0 / (ch*ch);
		}
	};
}


// single layer of an ANN
/*! \class NLLayer
	\brief Nonlinearity layer
	\details This layer applies a nonlinearity to all its inputs
*/
template<class T, template<typename> class A>
class NLLayer : public ILayer<T>
{
public:
	using typename ILayer<T>::WP_ILayer;
	using typename ILayer<T>::range_t;
	using typename ILayer<T>::const_range_t;
	
	NLLayer( unsigned neurons );
	
	// propagate signal through the layer
	void forward() override;
	void backward() override;
	
	// build up connections
	void setPreviousLayer( const WP_ILayer& prev ) override;
	void setNextLayer( const WP_ILayer& next ) override;
	
	// infos 
	unsigned getNumNeurons() const override { return mNumNeurons; };
	unsigned getNumInputs()  const override { return mNumNeurons; };
	
	const WP_ILayer& getPreviousLayer() const override { return mPreviousLayer; };
	const WP_ILayer& getNextLayer()     const override { return mNextLayer; };
	
private:
	
	// get access to layer data
	range_t getOutputMutable()   override { return range_t{mNeuronOut.data(),   mNeuronOut.data()   + mNeuronOut.size()};   };
	range_t getWeightsMutable()  override { return range_t{};                  												};
	range_t getBiasMutable()     override { return range_t{mBiasWeights.data(), mBiasWeights.data() + mBiasWeights.size()}; };
	range_t getGradientMutable() override { return range_t{mGradient.data(),    mGradient.data()    + mGradient.size()};    };
	
	unsigned mNumNeurons;
	
	// neurons
	std::vector<T> mNeuronOut;
	
	// weights
	std::vector<T> mBiasWeights;
	/// \todo maybe reintroduce weights to adapt steepness
	
	// error
	/// \todo allocate only when needed
	std::vector<T> mGradient;
	
	// connected layers
	WP_ILayer mPreviousLayer;
	WP_ILayer mNextLayer;
};

template<class T, template<typename> class A>
NLLayer<T, A>::NLLayer( unsigned neurons ) : 
	mNumNeurons( neurons ), 
	// allocate memory and zero initialize
	mNeuronOut( neurons ),
	mBiasWeights( neurons ),
	mGradient( neurons )
{
	
}

template<class T, template<typename> class A>
void NLLayer<T, A>::setPreviousLayer( const WP_ILayer& prev )
{
	// for now, do not allow to change the prev layer later on
	assert( mPreviousLayer.expired() );
	
	// check layer compatibility
	assert( prev.lock()->getNumNeurons() == getNumInputs() );
	
	mPreviousLayer = prev;
}

template<class T, template<typename> class A>
void NLLayer<T, A>::setNextLayer( const WP_ILayer& next )
{
	// for now, do not allow to change the prev layer later on
	assert( mNextLayer.expired() );
	
	// check layer compatibility
	assert( next.lock()->getNumInputs() == getNumNeurons() );
	
	mNextLayer = next;
}


template<class T, template<typename> class A>
void NLLayer<T, A>::forward()
{
	const_range_t input = mPreviousLayer.lock()->getOutput();
	
	// calculate new neuron sum
	auto trafo = A<T>();
	for(unsigned i = 0; i < mNumNeurons; ++i)
	{
		mNeuronOut[i] = trafo(input[i]);
	}
}


template<class T, template<typename> class A>
void NLLayer<T, A>::backward()
{
	// write the error into the previous layer's error var
	auto prevgrad = mNextLayer.lock()->getGradient();
	auto input = mPreviousLayer.lock()->getOutput();
	for(unsigned i = 0; i < mNumNeurons; ++i)
	{
		// apply activation function
		mGradient[i] = A<T>().deriv( input[i] ) * prevgrad[i];
	}
}


#endif // NL_LAYER_HPP_INCLUDED
