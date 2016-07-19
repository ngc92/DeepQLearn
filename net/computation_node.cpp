#include "computation_node.hpp"
#include "layer.hpp"

namespace net
{
void ComputationNode::backpropagate( const Vector& error, Solver& solver )
{
	if(mLayer)
	{
		mLayer->backward(error, mError, *this, solver);
		if(mSource)
		{
			mSource->backpropagate( mError, solver );
		}
	}
}
}
