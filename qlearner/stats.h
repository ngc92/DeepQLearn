#pragma once

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>

namespace qlearn
{
	using namespace boost::accumulators;
	
	class Stats
	{
	public:
		Stats(std::size_t window_size);
		
		float getSmoothReward() const;
		float getSmoothQVal() const;
		float getSmoothMSE() const;
		
		void record( float reward, float qval );
		void record_error( float error );
	private:
		// stats smoothed over last steps
		accumulator_set<float, stats<tag::rolling_mean> > mSmoothedReward;
		accumulator_set<float, stats<tag::rolling_mean> > mSmoothedQVal;
		accumulator_set<float, stats<tag::rolling_mean> > mSmoothedMSE;
	};
}
