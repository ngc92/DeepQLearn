#ifndef QLEARNER_HPP_INCLUDED
#define QLEARNER_HPP_INCLUDED

#include <memory>
#include <functional>
#include "qconfig.hpp"
#include "net/computation_graph.hpp"
#include "net/network.hpp"

namespace qlearn
{
	class QCore;
	class QLearner;
	
	using qlearn_callback = std::function<void(const QLearner& learner)>;
	
	class QLearner
	{
	public:
		QLearner( Config cfg, net::Network net );
		~QLearner();
		
		/// \todo this is a copy of the old learn_step function. Rework!
		// this puts a new learning step into the Q learner. It returns the
		// action ID that the Q-learner wants to test next.
		// gets current situation and reward that the last step generated.
		int learn_step( const Vector& situation, float reward, bool terminal, net::Solver& solver );
		
		const net::Network& network() const { return mNetwork; }
		
		void setCallback( qlearn_callback cb ) { mCallback = cb; };
	private:
		Config mConfig;
		std::unique_ptr<QCore> mCore;
		
		// the network setup
		net::Network mNetwork;
		net::ComputationGraph mNetworkGraph;
		
		net::Network mTargetNet;
		net::ComputationGraph mTargetGraph;
		
		qlearn_callback mCallback;
	};
}

#endif // QLEARNER_HPP_INCLUDED
