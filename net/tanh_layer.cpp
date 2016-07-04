#include "tanh_layer.hpp"
#include "solver.hpp"
#include <cmath>

namespace net
{
/// get the size of the layer output
std::size_t TanhLayer::getOutputSize() const
{
	return mBias.size();
}

void TanhLayer::process(const Vector& input, Vector& out) const
{
	out = (mBias + input).unaryExpr([](float x) { return std::tanh(x);} );
}

void TanhLayer::backward(const Vector& error, Vector& back, const ComputationNode& compute, Solver& solver) const
{
	auto deriv = [](number_t v) { return 1 - v*v; };
	back =  error.array() * (compute.output().unaryExpr(deriv)).array();
	solver(mBias, back);
}

void TanhLayer::update(Solver& solver)
{
	solver.update( mBias );
}

std::unique_ptr<ILayer> TanhLayer::clone() const
{
	return std::unique_ptr<ILayer>( new TanhLayer(*this) );
}
}
