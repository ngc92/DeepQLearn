#pragma once

#include "network.hpp"
#include <vector>
#include <algorithm>
#include <boost/range/algorithm/fill.hpp>

template<class T>
class WeightUpdater;

/// teacher class responsible for network learning based on batch weight updates.
/// it combines the network, the weight slopes and the update algorithm.
template<class T>
class BatchTeacher
{
public:
	BatchTeacher( std::shared_ptr<Network<T>> network, std::shared_ptr<WeightUpdater<T>> updater );
	
	/// performs a supervised learning sample
	void sample_supervised( const std::vector<T>& input, std::vector<T> desired_output );
	
	/// propagates the error signal backwards and records the training
	/// slopes. 
	void sample( const std::vector<T>& error );
	
	/// finishes a mini batch and changes weights.
	T finishMiniBatch();
	
	T getMSE() const;
	
	/// read access to weight slopes
	typedef boost::iterator_range<const T*> const_range_t;
	const_range_t getSlopes() const { return const_range_t{mWeightSlopes.data(), mWeightSlopes.data() + mWeightSlopes.size()};};
	
private:
	std::shared_ptr<Network<T>> mNetwork;
	std::shared_ptr<WeightUpdater<T>> mWeightUpdate;
	
	std::vector<T> mWeightSlopes;
	unsigned mSampleCount;
	
	// total squared error
	T mTotalSquaredError = 0;
};


template<class T>
BatchTeacher<T>::BatchTeacher( std::shared_ptr<Network<T>> network, std::shared_ptr<WeightUpdater<T>> updater ) : 
	mNetwork(network), 
	mWeightUpdate( updater ),
	mSampleCount(0)
{
	// calculate memory required for slope
	std::size_t mem = 0;
	for(unsigned i = 0; i < mNetwork->getNumLayers(); ++i)
	{
		mem += mNetwork->getLayer(i)->getWeights().size();
	}
	mWeightSlopes.resize(mem, T(0));
}


template<class T>
void BatchTeacher<T>::sample( const std::vector<T>& error )
{
	assert( error.size() == mNetwork->getNumOutputs() );
	for(auto& v : error)
		mTotalSquaredError += v*v;
	
	mNetwork->backward( error );
	T* pointer = mWeightSlopes.data();
	for(unsigned i = 0; i < mNetwork->getNumLayers(); ++i)
	{
		mNetwork->getLayer(i)->calcLearningSlopes( pointer );
	}
	assert( pointer == mWeightSlopes.data() + mWeightSlopes.size());
	mSampleCount++;
}

template<class T>
void BatchTeacher<T>::sample_supervised( const std::vector<T>& input, std::vector<T> desired_output )
{
	mNetwork->forward( input );
	auto out = mNetwork->getOutput();
	
	assert( desired_output.size() == out.size() );
	
	for(unsigned i = 0; i < desired_output.size(); ++i)
		desired_output[i] -= out[i];
	
	sample( desired_output );
}


template<class T>
T BatchTeacher<T>::finishMiniBatch( )
{
	// calculate mean
	T factor = T(1)/T(mSampleCount);
	std::for_each( mWeightSlopes.begin(), mWeightSlopes.end(), [factor](T& v) { v*= factor; } );
	
	mWeightUpdate->updateWeights( mWeightSlopes );
	
	// now set the new weights
	T* pointer = mWeightSlopes.data();
	for(unsigned i = 0; i < mNetwork->getNumLayers(); ++i)
	{
		mNetwork->getLayer(i)->updateWeights( pointer );
	}
	assert( pointer == mWeightSlopes.data() + mWeightSlopes.size());
	
	// reset
	boost::fill(mWeightSlopes, T(0));
	
	auto error = getMSE();
	
	mSampleCount = 0;
	mTotalSquaredError = 0;
	return error;
}

template<class T>
T BatchTeacher<T>::getMSE() const
{
	return mTotalSquaredError / mSampleCount / mNetwork->getNumOutputs();
}


// helper class: weight updater
template<class T>
class WeightUpdater
{
public:
	virtual ~WeightUpdater() {};
	
	// updates the vector slopes with the changes for the weights
	virtual void updateWeights( std::vector<T>& slopes ) = 0;
};

// simple implementation
template<class T>
class BatchWeightUpdater : public WeightUpdater<T>
{
public:
	BatchWeightUpdater( T learning_rate ) : mLearningRate(learning_rate)
	{
	}
	
	
	void updateWeights( std::vector<T>& slopes ) override;
private:
	T mLearningRate = 1.0;
};

template<class T>
void BatchWeightUpdater<T>::updateWeights( std::vector<T>& slopes )
{
	for(auto& v : slopes )
		v *= mLearningRate;
}
