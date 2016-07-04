
#pragma once

#include "layer.hpp"

namespace net
{
class ReLULayer : public ILayer
{
public:
	explicit ReLULayer(Matrix p) : mBias(p) {};

	/// get the size of the layer output
	std::size_t getOutputSize() const override;
	
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
