#include "layer.hpp"

namespace net
{
ComputationNode ILayer::forward( ComputationNode input ) const
{
	Vector output = process( input.output() );
	return ComputationNode( std::move(input), output, this );
}
}
