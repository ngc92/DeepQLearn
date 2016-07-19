#pragma once

#include "config.h"
#include <memory>

namespace net
{
class ComputationNode final
{
public:
	// ComputationNode at the beginning of the chain, just outputting the initial value
	explicit ComputationNode( Vector startvec ) :
		mSource( nullptr ),
		mOutput( std::move(startvec) ),
		mLayer( nullptr )
	{
	}

	ComputationNode( std::shared_ptr<ComputationNode> in, Vector out, const ILayer* l ) :
		/// \todo this is not nice, it requires us to do a dynamic memory allocation here.
		mSource( std::move(in) ),
		mOutput( std::move(out) ), mLayer( l )
	{
	}

	ComputationNode(ComputationNode&&) = default;
	ComputationNode& operator=(ComputationNode&&) = default;

	const Vector& input() const { return mSource->output(); };
	const Vector& output() const { return mOutput; };
	const Vector& error() const { return mError; };
	const ILayer* layer() const { return mLayer; };

	Vector& out_cache() { return mOutput; }
	
	// the interesting thing: backpropagate
	void backpropagate( const Vector& error, Solver& solver );

private:
	std::shared_ptr<ComputationNode> mSource;
	Vector mOutput;
	Vector mError;

	// layer that processed this node
	const ILayer* mLayer;
};
}
