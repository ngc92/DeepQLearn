#ifndef NETWORK_HPP_INCLUDED
#define NETWORK_HPP_INCLUDED

#include <memory>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>

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

template<class T>
class Network
{
	typedef std::shared_ptr<ILayer<T>> P_Layer;
	
public:
	Network( std::vector<int> layers );
	
private:
	unsigned mNumInputs;
	unsigned mNumOutputs;
	
	// the layers
	std::vector<P_Layer> mLayers;
	// have a separate reference to the input layer
	std::shared_ptr<InputLayer<T>> mInputLayer;
	
};


// most simple network setup
template<class T>
Network<T>::Network(std::vector<int> layers)
{
	assert( layers.size() >= 2 );
	
	mNumInputs = layers.front();
	mNumOutputs = layers.back();
	
	#ifndef NDEBUG
	std::cout << "creating network with " << layers.size() << " layers\n";
	std::cout << "  input: " << mNumInputs << "\n";
	#endif
	
	// generate input layer
	mInputLayer = std::make_shared<InputLayer<T>>( mNumInputs );
	
	// generate hidden layers
	for(unsigned i = 1; i < layers.size(); ++i)
	{
		auto layer = std::make_shared<Layer<T, typename activation::tanh<T>>>(layers.at(i-1), layers.at(i));
		mLayers.push_back(layer);
		#ifndef NDEBUG
		std::cout << "  layer: " << layers.at(i) << "neurons, 1 bias" << "\n";
		#endif
	}
	
	// connect the layers
	
	// initialize the weights
}

#endif // NETWORK_HPP_INCLUDED
