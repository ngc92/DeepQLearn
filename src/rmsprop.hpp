#pragma once


// simple implementation
template<class T>
class RMSPROPWeightUpdater : public WeightUpdater<T>
{
public:
	RMSPROPWeightUpdater( T learning_rate = 1.0, T RMSDecay = 0.9, T RMSLambda = 0.01 ) : 
		mLearningRate(learning_rate),
		mRMSDecay( RMSDecay ),
		mRMSLambda( RMSLambda )
	{
	}
	
	
	void updateWeights( std::vector<T>& slopes ) override;
private:
	T mLearningRate = 1.0;
	T mRMSDecay     = 0.9;
	T mRMSLambda    = 0.01;
	
	std::vector<T> mAverageGradient;
	std::vector<T> mAverageSQGrad;
};

template<class T>
void RMSPROPWeightUpdater<T>::updateWeights( std::vector<T>& slopes )
{
	if( mAverageGradient.size() == 0 )
	{
		mAverageGradient = slopes;
		mAverageSQGrad = slopes;
		for( auto& v : mAverageSQGrad) v *= v;
	}
	
	assert(mAverageGradient.size() == slopes.size());
	
	for( unsigned i = 0; i < slopes.size(); ++i)
	{
		auto slope = slopes[i];
		
		//slope -= /*weights_L2_norm*/0.001 * weights[i];
		
		mAverageGradient[i] = mRMSDecay * mAverageGradient[i] + (1.f - mRMSDecay) * slope;
		mAverageSQGrad[i]   = mRMSDecay * mAverageSQGrad[i] + (1.f - mRMSDecay) * slope * slope;
		
		auto denom = mAverageSQGrad[i] - mAverageGradient[i]*mAverageGradient[i] + mRMSLambda;

		slopes[i] = mLearningRate * slope / std::sqrt(denom);
	}
}
