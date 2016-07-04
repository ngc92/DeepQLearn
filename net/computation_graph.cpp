#include "computation_graph.h"
#include "computation_node.hpp"
#include "network.hpp"

namespace net
{
	ComputationGraph::ComputationGraph( const Network& network ) : 
		mInputNode( std::make_shared<ComputationNode>(Vector()) )
	{
		mLayers = network.getLayers();
	}
	
	using node_map = std::unordered_map<const ILayer*, std::shared_ptr<ComputationNode>>;
	
	std::shared_ptr<ComputationNode> getCompNode(node_map& nodes, const ILayer* layer, const std::shared_ptr<ComputationNode>& previous)
	{
		auto node = nodes.find( layer );
		if( node != nodes.end() )
		{
			return node->second;
		} else {
			auto inserted = nodes.emplace(layer, std::make_shared<ComputationNode>(previous, Vector(), layer));
			return inserted.first->second;
		}
	}
	
	const Vector& ComputationGraph::forward( const Vector& input )
	{
		mInputNode = std::make_shared<ComputationNode>(input);
		std::shared_ptr<ComputationNode> previous = mInputNode;

		for(const auto& layer : mLayers)
		{
			// check if we have a node for this layer
			std::shared_ptr<ComputationNode> target = getCompNode( mLayerNodeMap, layer.get(), previous );
			layer->forward( *previous, *target );
			previous = target;
		}
		
		mFinalNode = previous;
		
		return output();
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
