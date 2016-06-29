#include "network.hpp"

Network& Network::add_layer_imp( layer_t layer )
{
	mLayers.push_back( std::move(layer) );
	return *this;
}

ComputationNode Network::forward(const Vector& input) const
{
	ComputationNode node(input);

	for(auto& layer : mLayers)
	{
		node = layer->forward( std::move(node) );
	}
	return node;
}

ComputationNode Network::operator()( const Vector& input ) const
{
	return forward(input);
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
