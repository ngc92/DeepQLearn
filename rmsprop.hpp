#pragma once

#include "solver.hpp"
#include <unordered_map>

class RMSProp : public IUpdateRule
{
public:
	RMSProp(double lambda, double rate, double epsilon);

	void updateParameter(Matrix& parameter, const Matrix& gradient) override;

private:
	double lambda;
	double rate;
	double epsilon;

	Matrix& getRMS(const Matrix& parameter);
	const Matrix& updateRMS( const Matrix& parameter, const Matrix& gradient );

	std::unordered_map<const number_t*, Matrix> mRMS;
};
