#pragma once

#include "config.h"
#include "computation_node.hpp"

namespace net
{
class ILayer
{
public:
	/// get the size of the layer output
	virtual std::size_t getOutputSize() const = 0;
	
	/// call notation to pass a computation through this layer.
	/*ComputationNode operator()( ComputationNode input ) const
	{
		return forward( std::move(input) );
	};
	*/

	/// propagate a computation node through this layer.
	//ComputationNode forward( ComputationNode input ) const;
	
	/// propagate a computation node through this layer.
	void forward( const ComputationNode& input, ComputationNode& output ) const;

	/// propagates error backward, and uses solver to track gradient
	virtual void backward(const Vector& error, Vector& back, const ComputationNode& compute, Solver& solver) const = 0;

	/// update the parameters according to the solver.
	virtual void update(Solver& solver) = 0;
	
	/// creates a copy of this layer.
	virtual std::unique_ptr<ILayer> clone() const = 0;
private:
	/// propagates input forward and calculates output.
	virtual void process(const Vector& input, Vector& output) const = 0;
};
}
