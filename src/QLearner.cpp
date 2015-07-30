#include "QLearner.h"
#include "fann.h"
#include <iostream>

QLearner::QLearner(int input_size, int action_count, int memory_length):
	mInputSize(input_size),
	mNumActions(action_count),
	mLearningNetwork( nullptr )
{
	mMemory.set_capacity( memory_length );

	// initialize network
	mQNetwork = fann_create_standard(3, mInputSize, (mInputSize+mNumActions+40)/2/*, (mInputSize+2*mNumActions+40)/4*/, mNumActions);
	//fann_randomize_weights(mQNetwork, 0.0, 0.0001);
	//mNetwork = fann_create_from_file("neuralbot.net");
	//fann_init_weights_minmax(mQNetwork, 0.0, 1.0);
	//fann_set_activation_function_hidden(mQNetwork, FANN_LINEAR_PIECE);
	fann_set_activation_function_output(mQNetwork, FANN_LINEAR);
	fann_set_train_error_function(mQNetwork, FANN_ERRORFUNC_LINEAR);
	fann_set_activation_function_hidden(mQNetwork, FANN_SIGMOID_SYMMETRIC_STEPWISE);
}

void QLearner::start_learning()
{
	mRunLearning = true;
	mLearnStepCounter = 0;

	// start the threads
	mPreparationThread = std::thread([this](){ prepare_training();});
	mLearningThread = std::thread([this](){ learn_thread();});
}

void QLearner::load(std::string name)
{
	fann_destroy( mQNetwork );
	mQNetwork = fann_create_from_file((name+".net").c_str());
}

void QLearner::save(std::string file) const
{
    fann_save(mQNetwork, (file+".net").c_str());
}

QLearner::~QLearner()
{
	mRunLearning = false;
	mLearningThread.join();
	mPreparationThread.join();
	fann_destroy( mQNetwork );
}

double QLearner::getAverageEpisodeReward() const
{
	if(mCurNetTotalEpisodes == 0)
		return 0;

	return mCurNetTotalReward / mCurNetTotalEpisodes;
}

int QLearner::learn_step( const std::vector<float>& situation, float reward, bool terminal )
{
	mStepCounter++;
	if(mLearningNetwork != nullptr && mStepCounter % mNetUpdateFrq == 0)
	{
		std::lock_guard<std::mutex> lock(mLockNetwork);
		if( mCallback )
			mCallback(*this);

		// reset reward stats
		mCurNetTotalReward = 0;
		mCurNetTotalEpisodes = 0;

		fann_destroy(mQNetwork);
		mQNetwork = fann_copy(mLearningNetwork);
	}

	// wait if learning is too slow
	if(mStepCounter > mLearnStepCounter * mStepsPerBatch && mMemory.size() >  mInitMemoryPop)
	{
		std::this_thread::sleep_for( std::chrono::milliseconds(10));
	}


	mMemoryCache.future   =  situation;
	mMemoryCache.terminal =  terminal;
	mMemoryCache.reward   =  reward;
	mCurrenEpisodeReward  += reward;
	if( mMemoryCache.situation.size() != 0)
		push_memory();

	if( terminal )
	{
		mCurNetTotalReward += mCurrenEpisodeReward;
		mCurNetTotalEpisodes++;
		mCurrenEpisodeReward = 0;
	}

	int action = getAction( mQNetwork, situation, mCurrentQuality );
	mMemoryCache.situation = situation;

//	std::cout << mCurrentQuality << "\n";

	mAverageQuality = mFloatingMean * mAverageQuality + (1-mFloatingMean) * mCurrentQuality;

	// update the strategy: adapt epsilon
	if( mCurrentEpsilon > mFinalEpsilon )
		mCurrentEpsilon -= (1.0 - mFinalEpsilon) / mEpsilonSteps;

	// with certain probability choose a random action
	auto random_action = std::discrete_distribution<int>({1 - mCurrentEpsilon, mCurrentEpsilon});
	if( random_action(mRandom))
	{
		auto ind_dst = std::uniform_int_distribution<int>(0, mNumActions-1);
		action = ind_dst(mRandom);
	}

	mMemoryCache.action = action;
	return action;
}

float* QLearner::assess( const std::vector<float>& situation )
{
	return fann_run(mQNetwork, &situation.front() );
}

int QLearner::getAction( fann* network, const std::vector<float>& situation, float& quality )
{
	// greedy algorithm that generates the next action.
	float* scores = fann_run(network, &situation.front() );
	float* max = std::max_element(scores, scores + fann_get_num_output(network));
	quality = *max;
	return  max - scores;
}

void QLearner::push_memory()
{
	std::lock_guard<std::mutex> lock(mLockMemory);
	mMemory.push_back( mMemoryCache );
}

auto QLearner::get_memory( int index ) -> MemoryEntry
{
	std::lock_guard<std::mutex> lock(mLockMemory);
	return mMemory[index]; /// \todo this requires coying and memory allocation.
	/// not nice for performance
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// 					training preparation thread
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// helper function for gathering the data for a mini batch
void QLearner::build_mini_batch( fann* network, fann_train_data* data )
{
	for(unsigned i = 0; i < fann_length_train_data(data); ++i)
	{
		int sample = std::uniform_int_distribution<int>(0, mMemory.size() - 1)(mRandom);
		MemoryEntry trans = get_memory( sample );

		// best value that can be reached from here
		float y = 0;
		if( !trans.terminal )
		{
			int best = getAction(network, trans.future, y);
			// plus current reward
			y *= mDiscountFactor;
		}
		y += trans.reward;

		// get current output
		float* result = fann_run(network, trans.situation.data());
		// we are writing into the bowels of network here. should not be
		// dangerous though
		result[trans.action] = y;

		// copy to train data
		std::copy(trans.situation.begin(), trans.situation.end(), data->input[i]);
		for( unsigned j = 0; j < mNumActions; ++j)
		{
			data->output[i][j] = std::max(-1.f, std::min(1.f, result[j]));
		}

	}
}

void QLearner::prepare_training()
{
	// NN used for training data generation
	fann* prepare_ann = nullptr;
	// wait for data to become ready
	while(mRunLearning)
	{
		// check that we have enough data to base our training on
		if( mMemory.size() < mInitMemoryPop )
		{
			std::this_thread::sleep_for( std::chrono::milliseconds(10));
			continue;
		}

		// only prepare learning if we do not have enough data
		if(mPreprocessedTrainingData.size() < 100)
		{
			// update the training prep. ANN to the newest version.
			/// \todo but try to prevent copying for every data point!
			{
				if(prepare_ann)
					fann_destroy(prepare_ann);
				std::lock_guard<std::mutex> lock(mLockNetwork);
				prepare_ann = fann_copy( mQNetwork );
			}

			// generate new train data, either reusing old memory or allocating new
			fann_train_data* new_data = nullptr;
			if(mSpentTrainingData.empty())
			{
				// create training data
				new_data = fann_create_train(mMiniBatchSize, mInputSize, std::max(2, mNumActions) );
			}
			 else
			{
				// take training data from spent set
				std::lock_guard<std::mutex> lock(mSpentTrainingMutex);
				if(!mSpentTrainingData.empty())
				{
					new_data = mSpentTrainingData.front();
					mSpentTrainingData.pop_front();
				}
			}

			// build a new dataset
			build_mini_batch( prepare_ann, new_data );

			// add to data
			{
				std::lock_guard<std::mutex> lock(mTrainingDataMutex);
				mPreprocessedTrainingData.push_back( new_data );
			}
		}
	}
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -					learning thread								 -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void QLearner::learn_thread()
{
	mLearningNetwork = fann_copy(mQNetwork);
	while(mRunLearning)
	{
		// check that train data is present
		if(mPreprocessedTrainingData.empty())
		{
			std::this_thread::sleep_for( std::chrono::milliseconds(10));
			continue;
		}

		// grab new train data
		fann_train_data* training = nullptr;
		{
			std::lock_guard<std::mutex> lock(mTrainingDataMutex);
			// check again that we have data
			if(mPreprocessedTrainingData.empty())
			{
				std::this_thread::sleep_for( std::chrono::milliseconds(10));
				continue;
			}

			training = mPreprocessedTrainingData.front();
			mPreprocessedTrainingData.pop_front();
		}


		mLearnStepCounter++;
		{
			std::lock_guard<std::mutex> lock(mLearningMutex);
			// set learning parameters
			// do not change any weights for the first 1000 steps, use those to accumulate
			// gradient info to initialize the normalization.
			fann_set_learning_rate(mLearningNetwork, mLearnStepCounter < 1000 ? 0.f : mLearningRate);
			fann_set_training_algorithm( mLearningNetwork, FANN_TRAIN_RMSPROP);

			// train an epoch
			float error = fann_train_epoch(mLearningNetwork, training) * mNumActions;
			/// \todo ensure that error does not diverge
			mAverageError = mFloatingMean * mAverageError + (1-mFloatingMean)*error;

			{
				std::lock_guard<std::mutex> lock(mSpentTrainingMutex);
				mSpentTrainingData.push_back(training);
			}
		}
	}
}
