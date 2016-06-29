#pragma once

#include "config.h"
#include "computation_node.hpp"

enum class LayerType
{
	FullyConnected,
	Bias,
	ReLU,
};

class ILayer
{
	// clones this layer.
	//virtual std::shared_ptr<ILayer> clone() const = 0;
public:
	virtual LayerType getType() const = 0;

	ComputationNode operator()(ComputationNode input ) const
	{
		return forward( std::move(input) );
	};

	ComputationNode forward( ComputationNode input ) const;

	// propagates input forward and calculates output
	virtual Vector process(const Vector& input) const = 0;

	// propagates error backward, and uses solver to track gradient
	virtual Vector backward(const Vector& error, const ComputationNode& compute, Solver& solver) const = 0;

	// update the parameters
	virtual void update(Solver& solver) = 0;
	
	virtual std::unique_ptr<ILayer> clone() const = 0;
};
