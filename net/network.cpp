#include "network.hpp"

namespace net
{
Network& Network::add_layer_imp( layer_t layer )
{
	mLayers.push_back( std::move(layer) );
	return *this;
}

ComputationNode Network::forward(Vector input) const
{
	ComputationNode node( std::move(input) );

	for(auto& layer : mLayers)
	{
		node = layer->forward( std::move(node) );
	}
	return node;
}

ComputationNode Network::operator()( Vector input ) const
{
	return forward( std::move(input) );
}

void Network::update(Solver& solver)
{
    for(auto& l : mLayers)
	{
        l->update(solver);
	}
}

Network Network::clone() const
{
	Network newnet;
	for(const auto& layer : mLayers)
	{
		newnet.mLayers.push_back( layer->clone() );
	}
	return newnet;
}
}
