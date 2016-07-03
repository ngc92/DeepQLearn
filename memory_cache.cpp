#include "memory_cache.hpp"
#include <cassert>

MemoryCache::MemoryCache( std::size_t capacity )
{
	mMemory.set_capacity( capacity );
}

void MemoryCache::push( Experience entry )
{
	mMemory.push_back( std::move(entry) );  
}

void MemoryCache::push( Vector future, bool terminal, float reward )
{
	if(!mCachePrep) return;
	mMemoryCache.future = std::move(future);
	mMemoryCache.terminal = terminal;
	mMemoryCache.reward = reward;
	push( std::move(mMemoryCache) );
	mCachePrep = false;
}

void MemoryCache::prepare_next( Vector situation, int action )
{
	mMemoryCache.situation = std::move(situation);
	mMemoryCache.action = action;
	mCachePrep = true;
}

Experience MemoryCache::get( std::size_t index ) const
{
	assert(index  < mMemory.size());
	return mMemory[index];
}
