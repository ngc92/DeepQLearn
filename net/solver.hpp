#pragma once

#include "config.h"
#include <unordered_map>
#include <memory>

namespace net
{
	class SolverTestAccess;
	class IUpdateRule;

// non-polymorphic solver class that is responsible for general book keeping
class Solver final
{
	friend class SolverTestAccess;

public:
	Solver(std::unique_ptr<IUpdateRule>);

	template<class Value, class Expr>
	void operator()(const Value& val, Expr&& expr)
	{
		getGradient(val) += expr;
	}

	void update(Matrix& param);

	// const version to retrieve the gradient. Throws an exception, if
	// no gradient has been saved for value.
	const Matrix& getGradient( const Matrix& value ) const;

private:
	Matrix& getGradient( const Matrix& value );

	// internal function that accesses the buffer of gradients. param identifies the parameter whose gradient to
	// retrieve, whereas (rows, cols) specifies its size in case we need to insert zeros.
	Matrix& getGradientBuffer(const number_t* param, std::size_t rows, std::size_t cols);

	// remember all parameter gradients
	std::unordered_map<const number_t*, Matrix> mGradientMap;
	std::unique_ptr<IUpdateRule> mUpdateRule;
};


class IUpdateRule
{
	public:
		virtual ~IUpdateRule() {};
		virtual void updateParameter(Matrix& parameter, const Matrix& gradient) = 0;
};

}
