#include "network.hpp"
#include "layer.hpp"

Network& Network::add_layer_imp( layer_t layer )
{
	mLayers.push_back( layer );
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