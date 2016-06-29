#pragma once

#include "config.h"
#include "layer.hpp" // necessary, so default work
#include <vector>
#include <memory>

/*! \class Network
	\brief Simple Feed Forward Network.
	\details Currently represents a simple feed forward network of layers.
*/
class Network
{
	typedef std::unique_ptr<ILayer> layer_t;
public:
	Network() = default;
	~Network() = default;
	Network( Network&& ) = default;
	Network& operator=( Network&& ) = default;

	template<class T>
	Network& add_layer( T layer )
	{
		return add_layer_imp( std::make_unique<T>(std::move(layer)) );
	}

	template<class T>
	Network& operator<<( T&& layer )
	{
		return add_layer( std::move(layer) );
	}

	// processes the vector through the network
	ComputationNode forward(const Vector& input) const;
	ComputationNode operator()( const Vector& input ) const;

	// update all layers
	void update(Solver& solver);
	
	// creates a deep copy of the network
	Network clone() const;

private:
	Network& add_layer_imp( layer_t layer );

	std::vector<layer_t> mLayers;
};
