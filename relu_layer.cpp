#include "relu_layer.hpp"
#include "solver.hpp"

Vector ReLULayer::process(const Vector& input) const
{
    return (mBias + input).cwiseMax(0);
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
	return std::unique_ptr<ILayer>( new ReLULayer(*this) );
}
