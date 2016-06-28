#include "solver.hpp"

Solver::Solver(std::unique_ptr<IUpdateRule> rule ) : mUpdateRule( std::move(rule))
{

}

Matrix& Solver::getGradient( const Matrix& value )
{
	return getGradientBuffer(value.data(), value.rows(), value.cols());
}

const Matrix& Solver::getGradient( const Matrix& value ) const
{
	return mGradientMap.at(value.data());
}

Matrix& Solver::getGradientBuffer(const number_t* param, std::size_t rows, std::size_t cols)
{
    if(mGradientMap.count(param))
	{
		return mGradientMap.at(param);
	} else
	{
		auto res = mGradientMap.emplace(param, Matrix::Zero(rows, cols));
		assert(res.second);
		return res.first->second;
	}
}

void Solver::update(Matrix& param)
{
	mUpdateRule->updateParameter(param, getGradient(param));
	getGradient(param).setZero( param.rows(), param.cols() );
}

