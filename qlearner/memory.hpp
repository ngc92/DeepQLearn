#pragma once

#include "config.h"
#include <boost/circular_buffer.hpp>

namespace qlearn 
{
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
	void push( const Experience& entry );
	void push( Experience&& entry );
	
	// pushes a newly created experience while avoiding all unnecessary memory allocation
	void emplace( const Vector& situation, int action, const Vector& future, float reward, bool terminal );
	
	const Experience& get( std::size_t index ) const;
	
	template<class T>
	const Experience& get_random( T& random );
	
	std::size_t size() const { return mMemory.size(); }
private:
	boost::circular_buffer<Experience> mMemory;
	
	Experience mCache;
};


template<class T>
const Experience& MemoryCache::get_random( T& random )
{
	std::size_t sample = std::uniform_int_distribution<std::size_t>(0, mMemory.size() - 1)(random);
	return get(sample);
}
}
