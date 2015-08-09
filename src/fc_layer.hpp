#ifndef FC_LAYER_HPP_INCLUDED
#define FC_LAYER_HPP_INCLUDED

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
	};
}


// single layer of an ANN
template<class T, template<typename> class A>
class FCLayer : public ILayer<T>
{
public:
	using typename ILayer<T>::WP_ILayer;
	using typename ILayer<T>::range_t;
	using typename ILayer<T>::const_range_t;
	
	FCLayer( unsigned inputs, unsigned outputs );
	
	void setOutput(const T* out) override;
	void randomizeWeights( const std::function<T()>& distribution ) override;
	
	// propagate signal through the layer
	void forward() override;
	
	// build up connections
	void setPreviousLayer( const WP_ILayer& prev ) override;
	void setNextLayer( const WP_ILayer& next ) override;
	
	// infos 
	unsigned getNumNeurons() const override { return mNumNeurons; };
	unsigned getNumInputs()  const override { return mNumInputs; };
	
	const WP_ILayer& getPreviousLayer() const override { return mPreviousLayer; };
	const WP_ILayer& getNextLayer()     const override { return mNextLayer; };
	
	// get access to layer data
	const_range_t getOutput()   const override { return const_range_t{mNeuronOut.data(),   mNeuronOut.data()   + mNeuronOut.size()};   };
	const_range_t getNeuronIn() const override { return const_range_t{mNeuronSum.data(),   mNeuronSum.data()   + mNeuronSum.size()};   };
	const_range_t getWeights()  const override { return const_range_t{mWeights.data(),     mWeights.data()     + mWeights.size()};     };
	const_range_t getBias()     const override { return const_range_t{mBiasWeights.data(), mBiasWeights.data() + mBiasWeights.size()}; };
	const_range_t getError()    const override { return const_range_t{mError.data(),       mError.data()       + mError.size()};       };
	
private:
	
	unsigned mNumNeurons;
	unsigned mNumInputs;
	
	// neurons
	std::vector<T> mNeuronSum;
	std::vector<T> mNeuronOut;
	
	// weights
	std::vector<T> mWeights;
	std::vector<T> mBiasWeights;
	
	// error
	/// \todo allocate only when needed
	std::vector<T> mError;
	
	// connected layers
	WP_ILayer mPreviousLayer;
	WP_ILayer mNextLayer;
};

template<class T, template<typename> class A>
FCLayer<T, A>::FCLayer( unsigned inputs, unsigned outputs ) : 
	mNumNeurons( outputs ), 
	mNumInputs( inputs ),
	// allocate memory and zero initialize
	mNeuronSum( outputs ),
	mNeuronOut( outputs ),
	mWeights( inputs * outputs ),
	mBiasWeights( outputs ),
	mError( outputs )
{
	
}

template<class T, template<typename> class A>
void FCLayer<T, A>::setOutput(const T* out) 
{
	std::copy(out, out + mNumNeurons, mNeuronOut.begin());
}

template<class T, template<typename> class A>
void FCLayer<T, A>::randomizeWeights( const std::function<T()>& distribution )
{
	std::generate( mWeights.begin(), mWeights.end(), distribution );
}

template<class T, template<typename> class A>
void FCLayer<T, A>::setPreviousLayer( const WP_ILayer& prev )
{
	// for now, do not allow to change the prev layer later on
	assert( mPreviousLayer.expired() );
	
	// check layer compatibility
	assert( prev.lock()->getNumNeurons() == getNumInputs() );
	
	mPreviousLayer = prev;
}

template<class T, template<typename> class A>
void FCLayer<T, A>::setNextLayer( const WP_ILayer& next )
{
	// for now, do not allow to change the prev layer later on
	assert( mNextLayer.expired() );
	
	// check layer compatibility
	assert( next.lock()->getNumInputs() == getNumNeurons() );
	
	mNextLayer = next;
}


template<class T, template<typename> class A>
void FCLayer<T, A>::forward()
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
		
		// BIAS neuron
		tempsum += mBiasWeights[i];
		
		mNeuronSum[i] = tempsum;
		mNeuronOut[i] = tempsum;
	}
	
	// apply activation function.
	// TODO is this better here or in the other loop?
	std::transform(mNeuronOut.begin(), mNeuronOut.end(), mNeuronOut.begin(), A<T>());
}




#endif // FC_LAYER_HPP_INCLUDED
