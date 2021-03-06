#ifndef INPUT_LAYER_HPP_INCLUDED
#define INPUT_LAYER_HPP_INCLUDED

#include "layer.hpp"

// class for the input layer
template<class T>
class InputLayer : public ILayer<T>
{
public:
	using typename ILayer<T>::WP_ILayer;
	using typename ILayer<T>::range_t;
	using typename ILayer<T>::const_range_t;
	
	
	InputLayer( unsigned neurons );
	
	// propagate signal through the layer
	void forward() override;
	void backward() override;
	void calcLearningSlopes( T*& target ) override {};
	std::shared_ptr<ILayer<T>> clone() const override { return std::make_shared<InputLayer<T>>( *this );};
	
	// build up connections
	void setPreviousLayer( const WP_ILayer& prev ) override { assert(0); };
	void setNextLayer( const WP_ILayer& next ) override;
	
	// infos 
	unsigned getNumNeurons() const override   { return mNumNeurons; };
	unsigned getNumInputs()  const override   { return 0; };
	const char* getLayerType() const override { return "input"; };
	
	const WP_ILayer& getPreviousLayer() const override { return EmptyPrevious; };
	const WP_ILayer& getNextLayer()     const override { return mNextLayer; };
	
private:
	
	// get access to layer data
	range_t getOutputMutable()   override { return range_t{mNeuronOut.data(),   mNeuronOut.data()   + mNeuronOut.size()};   };
	range_t getWeightsMutable()  override { return range_t{}; };
	range_t getGradientMutable()    override { return range_t{}; };
	
	
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
	// check layer compatibility
	assert( next.lock()->getNumInputs() == getNumNeurons() );
	
	mNextLayer = next;
}

template<class T>
void InputLayer<T>::forward()
{
}

template<class T>
void InputLayer<T>::backward()
{
}

template<typename T>
const typename ILayer<T>::WP_ILayer InputLayer<T>::EmptyPrevious{};


#endif // INPUT_LAYER_HPP_INCLUDED
