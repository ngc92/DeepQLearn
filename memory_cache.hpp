#pragma once

#include "config.h"
#include <boost/circular_buffer.hpp>

struct Experience
{
	Vector situation;
	int action;
	Vector future;
	float reward;
	bool terminal;
};

class MemoryCache
{
public:
	MemoryCache( std::size_t capacity );
	
	// building up the memory
	// push a complete memory
	void push( Experience entry );
	// finishes the cached state, 
	void push( Vector future, bool terminal, float reward );
	void prepare_next( Vector situation, int action );
	
	Experience get( std::size_t index ) const;
	
	template<class T>
	Experience get_random( T& random );
	
	std::size_t size() const { return mMemory.size(); }
private:
	boost::circular_buffer<Experience> mMemory;
	
	Experience mMemoryCache;
	bool mCachePrep = false;
};


template<class T>
Experience MemoryCache::get_random( T& random )
{
	std::size_t sample = std::uniform_int_distribution<std::size_t>(0, mMemory.size() - 1)(random);
	return get(sample);
}
