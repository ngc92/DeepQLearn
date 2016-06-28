
#pragma once

#include "layer.hpp"

class FcLayer : public ILayer
{
public:
	explicit FcLayer(Matrix p) : mMatrix(p) {};

	LayerType getType() const override { return LayerType::FullyConnected; };

	const Matrix& getParameter() const { return mMatrix; };

	// propagates input forward and calculates output
	Vector process(const Vector& input) const override;

	// propagates error backward, and uses solver to track gradient
	Vector backward(const Vector& error, const ComputationNode& compute, Solver& solver) const override;

	void update(Solver& solver) override;
private:
	Matrix mMatrix;
};
