#ifndef LAYER_HPP_INCLUDED
#define LAYER_HPP_INCLUDED

#include <vector>
#include <algorithm>

// base class of all layers
template<class T>
class ILayer
{
public:
	typedef std::weak_ptr<const ILayer>  WP_ILayer;
	
	virtual ~ILayer() {};
	virtual void forward() = 0;
	
	void setOutput(const std::vector<T>& container)
	{
		assert( container.size() == getNumNeurons() );
		setOutput( container.data() );
	}
	
	virtual void setOutput(const T* out) = 0; // forces the layers output to be \p out. A safer version (overload) of this function is provided 
											  // that checks length compatibility
	
	// build up connections
	virtual void setPreviousLayer( const WP_ILayer& prev ) = 0;
	virtual void setNextLayer( const WP_ILayer& next ) = 0;
	
	// infos
	virtual unsigned getNumNeurons() const = 0;
	virtual unsigned getNumInputs()  const = 0;
	
	virtual const WP_ILayer& getPreviousLayer() const = 0;
	virtual const WP_ILayer& getNextLayer()     const = 0;
	
	// get access to layer data
	virtual const T* getOutput()   const = 0;
	virtual const T* getNeuronIn() const = 0; 
	virtual const T* getWeights()  const = 0;
	virtual const T* getBias()     const = 0;

};

// single layer of an ANN
template<class T, typename A>
class Layer : public ILayer<T>
{
public:
	using typename ILayer<T>::WP_ILayer;
	
	Layer( unsigned inputs, unsigned outputs );
	
	void setOutput(const T* out) override;
	
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
	const T* getOutput() 	 const override { return mNeuronOut.data(); };
	const T* getNeuronIn()   const override { return mNeuronSum.data(); };
	const T* getWeights()    const override { return mWeights.data(); };
	const T* getBias()       const override { return mBiasWeights.data(); };
	
private:
	
	unsigned mNumNeurons;
	unsigned mNumInputs;
	
	// neurons
	std::vector<T> mNeuronSum;
	std::vector<T> mNeuronOut;
	
	// weights
	std::vector<T> mWeights;
	std::vector<T> mBiasWeights;
	
	// connected layers
	WP_ILayer mPreviousLayer;
	WP_ILayer mNextLayer;
};

template<class T, class A>
Layer<T, A>::Layer( unsigned inputs, unsigned outputs ) : 
	mNumNeurons( outputs ), 
	mNumInputs( inputs ),
	// allocate memory and zero initialize
	mNeuronSum( outputs ),
	mNeuronOut( outputs ),
	mWeights( inputs * outputs ),
	mBiasWeights( outputs )
{
	
}

template<class T, typename A>
void Layer<T, A>::setOutput(const T* out) 
{
	std::copy(out, out + mNumNeurons, mNeuronOut.begin());
}

template<class T, typename A>
void Layer<T, A>::setPreviousLayer( const WP_ILayer& prev )
{
	// for now, do not allow to change the prev layer later on
	assert( mPreviousLayer.expired() );
	
	// check layer compatibility
	assert( prev.lock()->getNumNeurons() == getNumInputs() );
	
	mPreviousLayer = prev;
}

template<class T, typename A>
void Layer<T, A>::setNextLayer( const WP_ILayer& next )
{
	// for now, do not allow to change the prev layer later on
	assert( mNextLayer.expired() );
	
	// check layer compatibility
	assert( next.lock()->getNumInputs() == getNumNeurons() );
	
	mNextLayer = next;
}


template<class T, typename A>
void Layer<T, A>::forward()
{
	const T* input = mPreviousLayer.lock()->getOutput();
	
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
	std::transform(mNeuronOut.begin(), mNeuronOut.end(), mNeuronOut.begin(), A());
}


// class for the input layer
template<class T>
class InputLayer : public ILayer<T>
{
public:
	using typename ILayer<T>::WP_ILayer;
	
	InputLayer( unsigned neurons );
	
	void setOutput(const T* out) override;
	
	// propagate signal through the layer
	void forward() override;
	
	// build up connections
	void setPreviousLayer( const WP_ILayer& prev ) override { assert(0); };
	void setNextLayer( const WP_ILayer& next ) override;
	
	// infos 
	unsigned getNumNeurons() const override { return mNumNeurons; };
	unsigned getNumInputs()  const override { return 0; };
	
	const WP_ILayer& getPreviousLayer() const override { return EmptyPrevious; };
	const WP_ILayer& getNextLayer()     const override { return mNextLayer; };
	
	// get access to layer data
	const T* getOutput() 	 const override { return mNeuronOut.data(); };
	const T* getNeuronIn()   const override { return nullptr; };
	const T* getWeights()    const override { return nullptr; };
	const T* getBias()       const override { return nullptr; };
	
private:
	static const WP_ILayer EmptyPrevious;
	
	unsigned mNumNeurons;
	
	// neurons
	std::vector<T> mNeuronOut;
	
	// connected layers
	WP_ILayer mNextLayer;
};

template<class T>
InputLayer<T>::InputLayer( unsigned neurons ) : 
	mNumNeurons( neurons ), 
	// allocate memory and zero initialize
	mNeuronOut( neurons )
{
	
}

template<class T>
void InputLayer<T>::setNextLayer( const WP_ILayer& next )
{
	// for now, do not allow to change the prev layer later on
	assert( mNextLayer.expired() );
	
	// check layer compatibility
	assert( next.lock()->getNumInputs() == getNumNeurons() );
	
	mNextLayer = next;
}

template<class T>
void InputLayer<T>::setOutput(const T* out) 
{
	std::copy(out, out + mNumNeurons, mNeuronOut.begin());
}


template<class T>
void InputLayer<T>::forward()
{
}


#endif // LAYER_HPP_INCLUDED
