#include "relu_layer.hpp"
#include "solver.hpp"

namespace net
{
/// get the size of the layer output
std::size_t ReLULayer::getOutputSize() const
{
	return mBias.size();
}

void ReLULayer::process(const Vector& input, Vector& out) const
{
	out = (mBias + input).cwiseMax(0);
}

void ReLULayer::backward(const Vector& error, Vector& back, const ComputationNode& compute, Solver& solver) const
{
	auto deriv = [](number_t v) {return v > 0 ? 1 : 0;};
	back = error.array() * (compute.output().unaryExpr(deriv)).array();
	solver(mBias, back);
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
