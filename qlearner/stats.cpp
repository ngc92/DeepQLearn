#include "stats.h"

namespace qlearn
{
	Stats::Stats(std::size_t window_size) : 
		mSmoothedReward( tag::rolling_window::window_size = 10000 ),
		mSmoothedQVal( tag::rolling_window::window_size = 10000 ),
		mSmoothedMSE( tag::rolling_window::window_size = 10000 )
	{
		
	}
		
	float Stats::getSmoothReward() const
	{
		return rolling_mean(mSmoothedReward);
	}
	float Stats::getSmoothQVal() const
	{
		return rolling_mean(mSmoothedQVal);
	}
	float Stats::getSmoothMSE() const
	{
		return rolling_mean(mSmoothedMSE);
	}

	void Stats::record( float reward, float qval )
	{
		mSmoothedReward(reward);
		mSmoothedQVal(qval);
	}
	
	void Stats::record_error( float error )
	{
		mSmoothedMSE(error);
	}
}

