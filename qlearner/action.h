#pragma once

#include "config.h"

namespace net
{
	class ComputationGraph;
}


namespace qlearn 
{
	struct Action
	{
		std::size_t id;
		float score;
	};
	
	Action getAction(net::ComputationGraph& graph, const Vector& situation);
}
