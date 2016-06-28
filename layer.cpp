#include "layer.hpp"

ComputationNode ILayer::forward( ComputationNode input ) const
{
	Vector output = process( input.output() );
	return ComputationNode( std::move(input), output, this );
}
