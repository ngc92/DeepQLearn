#include "computation_node.hpp"
#include "layer.hpp"

namespace net
{
void ComputationNode::backpropagate( const Vector& error, Solver& solver )
{
	if(mLayer)
	{
		mError = mLayer->backward(error, *this, solver);
		if(mSource)
		{
			mSource->backpropagate( mError, solver );
		}
	}
}
}
