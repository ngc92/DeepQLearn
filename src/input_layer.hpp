#ifndef INPUT_LAYER_HPP_INCLUDED
#define INPUT_LAYER_HPP_INCLUDED

#include "layer.hpp"

// class for the input layer
template<class T>
class InputLayer : public ILayer<T>
{
public:
	using typename ILayer<T>::WP_ILayer;
	
	InputLayer( unsigned neurons );
	
	void setOutput(const T* out) override;
	
	void randomizeWeights( const std::function<T()>& distribution ) override { /* no weights here */ }
	
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
	const T* getError()      const override { return nullptr; };
	
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

template<typename T>
const typename ILayer<T>::WP_ILayer InputLayer<T>::EmptyPrevious{};


#endif // INPUT_LAYER_HPP_INCLUDED
