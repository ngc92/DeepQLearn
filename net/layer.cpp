#include "layer.hpp"

namespace net
{
void ILayer::forward( const ComputationNode& input, ComputationNode& output ) const
{
	process( input.output(), output.out_cache() );
}
}
