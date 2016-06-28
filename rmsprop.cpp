#include "rmsprop.hpp"
#include <cassert>

RMSProp::RMSProp(double l, double r, double e) : lambda(l), rate(r), epsilon(e)
{
}

Matrix& RMSProp::getRMS(const Matrix& parameter)
{
	auto param = parameter.data();
	if(mRMS.count(param))
	{
		return mRMS.at(param);
	} else
	{
		auto res = mRMS.emplace(param, parameter.array() * parameter.array());
		assert(res.second);
		return res.first->second;
	}
}

const Matrix& RMSProp::updateRMS( const Matrix& parameter, const Matrix& gradient )
{
	Matrix& rms = getRMS( parameter );
	rms *= lambda;
	rms += (1-lambda) * (gradient.array() * gradient.array()).matrix();
	return rms;
}

void RMSProp::updateParameter(Matrix& parameter, const Matrix& gradient)
{
	auto rms = updateRMS(parameter, gradient);
	parameter -= (rate * gradient.array() / sqrt(rms.array() + epsilon)).matrix();
}

