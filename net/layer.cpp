#include "layer.hpp"

namespace net
{
/*ComputationNode ILayer::forward( ComputationNode input ) const
{
	Vector output;
	process( input.output(), output );
	return ComputationNode( std::move(input), output, this );
}
*/
void ILayer::forward( const ComputationNode& input, ComputationNode& output ) const
{
	process( input.output(), output.out_cache() );
}
}
