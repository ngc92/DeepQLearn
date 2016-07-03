#ifndef COMPUTATION_GRAPH_HPP_INCLUDED
#define COMPUTATION_GRAPH_HPP_INCLUDED

#include <unordered_map>

namespace net
{
	class Solver;
	class ILayer;
	
	class ComputationGraph
	{
	public:
		ComputationGraph();
		void forward( const Network& net, const Vector& input );
		void backpropagate( const Vector& error, Solver& solver );
		void clear();
		
		// get computation results
		const Vector& output() const;
	private:
		ComputationNode mInputNode;
		ComputationNode* mFinalNode;
		std::unordered_map<ILayer*, ComputationNode> mLayerNodeMap;
	};
}

#endif // COMPUTATION_GRAPH_HPP_INCLUDED
