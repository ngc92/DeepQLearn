#ifndef COMPUTATION_GRAPH_H_INCLUDED
#define COMPUTATION_GRAPH_H_INCLUDED

#include <unordered_map>
#include <memory>
#include "config.h"

namespace net
{
	class Solver;
	class ILayer;
	class Network;
	class ComputationNode;
	
	class ComputationGraph
	{
	public:
		ComputationGraph() = default;
		ComputationGraph( const Network& net );
		const Vector& forward( const Vector& input );
		void backpropagate( const Vector& error, Solver& solver );
		void clear();
		
		// get computation results
		const Vector& output() const;
	private:
		std::shared_ptr<ComputationNode> mInputNode;
		std::shared_ptr<ComputationNode> mFinalNode;
		std::unordered_map<const ILayer*, std::shared_ptr<ComputationNode>> mLayerNodeMap;
		std::vector<std::shared_ptr<ILayer>> mLayers;
	};
}


#endif // COMPUTATION_GRAPH_H_INCLUDED
