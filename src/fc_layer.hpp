#ifndef FC_LAYER_HPP_INCLUDED
#define FC_LAYER_HPP_INCLUDED

// Fully Connected Layer 
#include "layer.hpp"

// single layer of an ANN
template<class T>
class FCLayer : public ILayer<T>
{
public:
	using typename ILayer<T>::WP_ILayer;
	using typename ILayer<T>::range_t;
	using typename ILayer<T>::const_range_t;
	
	FCLayer( unsigned inputs, unsigned outputs );
	
	// propagate signal through the layer
	void forward() override;
	void backward() override;
	void calcLearningSlopes( T*& target ) override;
	
	std::shared_ptr<ILayer<T>> clone() const override { return std::make_shared<FCLayer<T>>( *this );};
	
	// build up connections
	void setPreviousLayer( const WP_ILayer& prev ) override;
	void setNextLayer( const WP_ILayer& next ) override;
	
	// infos 
	unsigned getNumNeurons() const override   { return mNumNeurons; };
	unsigned getNumInputs()  const override   { return mNumInputs;  };
	const char* getLayerType() const override { return "fc";        };
	
	const WP_ILayer& getPreviousLayer() const override { return mPreviousLayer; };
	const WP_ILayer& getNextLayer()     const override { return mNextLayer; };
	
private:
	
	// get access to layer data
	range_t getOutputMutable()   override { return range_t{mNeuronOut.data(),   mNeuronOut.data()   + mNeuronOut.size()};   };
	range_t getWeightsMutable()  override { return range_t{mWeights.data(),     mWeights.data()     + mWeights.size()};     };
	range_t getGradientMutable()    override { return range_t{mGradient.data(),       mGradient.data()       + mGradient.size()};       };
	
	unsigned mNumNeurons;
	unsigned mNumInputs;
	
	// neurons
	std::vector<T> mNeuronOut;
	
	// weights
	std::vector<T> mWeights;
	
	// error
	/// \todo allocate only when needed
	std::vector<T> mGradient;
	
	// connected layers
	WP_ILayer mPreviousLayer;
	WP_ILayer mNextLayer;
};

template<class T>
FCLayer<T>::FCLayer( unsigned inputs, unsigned outputs ) : 
	mNumNeurons( outputs ), 
	mNumInputs( inputs ),
	// allocate memory and zero initialize
	mNeuronOut( outputs ),
	mWeights( inputs * outputs ),
	mGradient( inputs )
{
	
}

template<class T>
void FCLayer<T>::setPreviousLayer( const WP_ILayer& prev )
{
	// for now, do not allow to change the prev layer later on
	assert( mPreviousLayer.expired() );
	
	// check layer compatibility
	assert( prev.lock()->getNumNeurons() == getNumInputs() );
	
	mPreviousLayer = prev;
}

template<class T>
void FCLayer<T>::setNextLayer( const WP_ILayer& next )
{
	// for now, do not allow to change the prev layer later on
	assert( mNextLayer.expired() );
	
	// check layer compatibility
	assert( next.lock()->getNumInputs() == getNumNeurons() );
	
	mNextLayer = next;
}


template<class T>
void FCLayer<T>::forward()
{
	const_range_t input = mPreviousLayer.lock()->getOutput();
	
	// calculate new neuron sum
	for(unsigned i = 0; i < mNumNeurons; ++i)
	{
		T tempsum = 0;
		for(unsigned j = 0; j < mNumInputs; ++j)
		{
			tempsum += input[j] * mWeights[i * mNumInputs + j];
		}
		
		mNeuronOut[i] = tempsum;
	}
}


template<class T>
void FCLayer<T>::backward()
{
	// write the error into the previous layer's error var
	auto nextgrad = mNextLayer.lock()->getGradient();
	
	for(unsigned i = 0; i < mNumNeurons; ++i)
	{
		for(unsigned j = 0; j < mNumInputs; ++j)
			mGradient[j] += nextgrad[i] * mWeights[i * mNumInputs + j];
	}
}


template<class T>
void FCLayer<T>::calcLearningSlopes( T*& target )
{
	auto nextgrad = mNextLayer.lock()->getGradient();
	const_range_t input = mPreviousLayer.lock()->getOutput();
	
	for(unsigned i = 0; i < mNumNeurons; ++i)
	{
		for(unsigned j = 0; j < mNumInputs; ++j)
			target[i * mNumInputs + j] += input[j] * nextgrad[i];
	}
	
	target += mNumInputs * mNumNeurons;
}


#endif // FC_LAYER_HPP_INCLUDED
