#pragma once

#include "config.h"
#include "computation_node.hpp"

namespace net
{
class ILayer
{
public:
	/// call notation to pass a computation through this layer.
	ComputationNode operator()(ComputationNode input ) const
	{
		return forward( std::move(input) );
	};

	/// propagate a computation node through this layer.
	ComputationNode forward( ComputationNode input ) const;

	/// propagates error backward, and uses solver to track gradient
	virtual Vector backward(const Vector& error, const ComputationNode& compute, Solver& solver) const = 0;

	/// update the parameters according to the solver.
	virtual void update(Solver& solver) = 0;
	
	/// creates a copy of this layer.
	virtual std::unique_ptr<ILayer> clone() const = 0;
private:
	/// propagates input forward and calculates output.
	virtual Vector process(const Vector& input) const = 0;
};
}
