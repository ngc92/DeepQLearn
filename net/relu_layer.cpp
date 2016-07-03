#include "relu_layer.hpp"
#include "solver.hpp"

namespace net
{
void ReLULayer::process(const Vector& input, Vector& out) const
{
	out = (mBias + input).cwiseMax(0);
}

Vector ReLULayer::backward(const Vector& error, const ComputationNode& compute, Solver& solver) const
{
	auto deriv = [](number_t v) {return v > 0 ? 1 : 0;};
	Matrix E =  error.array() * (compute.output().unaryExpr(deriv)).array();
	solver(mBias, E);
	return E;
}

void ReLULayer::update(Solver& solver)
{
	solver.update( mBias );
}

std::unique_ptr<ILayer> ReLULayer::clone() const
{
	return std::make_unique<ReLULayer>(*this);
}

}
