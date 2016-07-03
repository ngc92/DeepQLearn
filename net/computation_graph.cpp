#include "computation_graph.h"
#include "network.hpp"

namespace net
{
	ComputationGraph::ComputationGraph( const Network& network ) : 
		mInputNode( Vector() ),
	{
		mLayers = network.getLayers();
	}
	
	void ComputationGraph::forward( const Vector& input )
	{
		const auto& layers = net.getLayers();
		mInputNode.output_cache() = input;
		ComputationNode* previous = &mInputNode;

		for(const auto& layer : layers)
		{
			// check if we have a node for this layer
			auto node = mLayerNodeMap.find( layer.get() );
			ComputationNode* target = nullptr;
			if( node != mLayerNodeMap.end() )
			{
				target = &node->second;
			} else {
				auto inserted = mLayerNodeMap.emplace(layer.get(), ComputationNode(previous, layer.get()));
				target = &inserted.first->second;
			}
			layer->forward( *target );
			previous = target;
		}
		
		mFinalNode = previous;
	}
	
	void ComputationGraph::clear()
	{
		mLayerNodeMap.clear();
	}
	
	const Vector& ComputationGraph::output() const
	{
		return mFinalNode->output();
	}
	
	void ComputationGraph::backpropagate( const Vector& error, Solver& solver )
	{
		mFinalNode->backpropagate(error, solver);
	}
}
