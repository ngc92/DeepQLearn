#include "tanh_layer.hpp"
#include "solver.hpp"
#include <cmath>

namespace net
{
Vector TanhLayer::process(const Vector& input) const
{
    return (mBias + input).unaryExpr([](float x) { return std::tanh(x);} );
}

Vector TanhLayer::backward(const Vector& error, const ComputationNode& compute, Solver& solver) const
{
	auto deriv = [](number_t v) { return 1 - v*v; };
	Matrix E =  error.array() * (compute.output().unaryExpr(deriv)).array();
	solver(mBias, E);
	return E;
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
