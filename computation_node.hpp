#pragma once

#include "config.h"
#include <memory>

class ComputationNode final
{
public:
	// ComputationNode at the beginning of the chain, just outputting the initial value
	explicit ComputationNode( Vector startvec ) :
		mSource( nullptr ),
		mOutput( startvec ),
		mLayer( nullptr )
	{
	}

	ComputationNode( ComputationNode in, Vector out, const ILayer* l ) :
		/// \todo this is not nice, it requires us to do a dynamic memory allocation here.
		mSource( std::unique_ptr<ComputationNode>( new ComputationNode(std::move(in)) ) ),
		mOutput( out ), mLayer( l )
	{
	}

	ComputationNode(ComputationNode&&) = default;
	ComputationNode& operator=(ComputationNode&&) = default;

	const Vector& input() const { return mSource->output(); };
	const Vector& output() const { return mOutput; };
	const Vector& error() const { return mError; };
	const ILayer* layer() const { return mLayer; };

	// the interesting thing: backpropagate
	void backpropagate( const Vector& error, Solver& solver );

private:
	std::unique_ptr<ComputationNode> mSource;
	Vector mOutput;
	Vector mError;

	// layer that processed this node
	const ILayer* mLayer;
};
