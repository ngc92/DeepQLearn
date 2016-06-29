#include "fc_layer.hpp"
#include "solver.hpp"
#include <iostream>

Vector FcLayer::process(const Vector& input) const
{
    return mMatrix * input;
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
	return std::unique_ptr<ILayer>( new FcLayer(*this) );
}
