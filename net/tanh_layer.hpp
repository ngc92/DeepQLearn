#pragma once

#include "layer.hpp"

namespace net
{
class TanhLayer : public ILayer
{
public:
	explicit TanhLayer(Matrix p) : mBias( std::move(p) ) {};

	const Matrix& getParameter() const { return mBias; };

	// propagates input forward and calculates output
	void process(const Vector& input, Vector& out) const override;

	// propagates error backward, and uses solver to track gradient
	Vector backward(const Vector& error, const ComputationNode& compute, Solver& solver) const override;

	void update(Solver& solver) override;
	
	std::unique_ptr<ILayer> clone() const override;
private:
	Matrix mBias;
};
}
