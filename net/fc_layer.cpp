#include "fc_layer.hpp"
#include "solver.hpp"
#include <iostream>

namespace net
{
void FcLayer::process(const Vector& input, Vector& out) const
{
	out.noalias() = mMatrix * input;
}

Vector FcLayer::backward(const Vector& error, const ComputationNode& compute, Solver& solver) const
{
	solver(mMatrix, error * compute.input().transpose());
	return mMatrix.transpose() * error;
}

void FcLayer::update(Solver& solver)
{
	solver.update( mMatrix );
}

std::unique_ptr<ILayer> FcLayer::clone() const
{
	return std::make_unique<FcLayer>( *this );
}
}
