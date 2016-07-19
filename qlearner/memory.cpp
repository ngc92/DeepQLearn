#include "memory.hpp"
#include <cassert>
#include <iostream>

namespace qlearn 
{
MemoryCache::MemoryCache( std::size_t capacity )
{
	mMemory.set_capacity( capacity );
}

void MemoryCache::push( const Experience& entry )
{
	mMemory.push_back( entry );  
}

void MemoryCache::push( Experience&& entry )
{
	mMemory.push_back( entry );  
}

void MemoryCache::emplace( const Vector& situation, int action, const Vector& future, float reward, bool terminal )
{
	// write to cache first. This copies, but does not allocate
	mCache.situation = situation;
	mCache.action = action;
	mCache.future = future;
	mCache.reward = reward;
	mCache.terminal = terminal;
	// this again copies, but does not allocate if memory is full
	mMemory.push_back( mCache );
}

const Experience& MemoryCache::get( std::size_t index ) const
{
	assert(index  < mMemory.size());
	return mMemory[index];
}
}
