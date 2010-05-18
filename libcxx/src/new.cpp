//===--------------------------- new.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h> 
#include <cxxabi.h> 

#include "new"


#if __APPLE__
    // On Darwin, there are two STL shared libraries and a lower level ABI
	// shared libray.  The global holding the current new handler is
    // in the ABI library and named __cxa_new_handler.
    #define __new_handler __cxxabiapple::__cxa_new_handler
#else
	static std::new_handler __new_handler;
#endif


// Implement all new and delete operators as weak definitions
// in this shared library, so that they can be overriden by programs
// that define non-weak copies of the functions.


__attribute__((__weak__, __visibility__("default")))
void *
operator new(std::size_t size) throw (std::bad_alloc)
{
    if (size == 0)
        size = 1;
    void* p;
    while ((p = ::malloc(size)) == 0)
    {
		// If malloc fails and there is a new_handler, 
		// call it to try free up memory.
        if (__new_handler)
            __new_handler();
        else
            throw std::bad_alloc();
    }
    return p;
}

__attribute__((__weak__, __visibility__("default")))
void*
operator new(size_t size, const std::nothrow_t&) throw()
{
    void* p = 0;
    try
    {
        p = ::operator new(size);
    }
    catch (...)
    {
    }
    return p;
}

__attribute__((__weak__, __visibility__("default")))
void*
operator new[](size_t size) throw (std::bad_alloc)
{
    return ::operator new(size);
}

__attribute__((__weak__, __visibility__("default")))
void*
operator new[](size_t size, const std::nothrow_t& nothrow) throw()
{
    void* p = 0;
    try
    {
        p = ::operator new[](size);
    }
    catch (...)
    {
    }
    return p;
}

__attribute__((__weak__, __visibility__("default")))
void
operator delete(void* ptr) throw ()
{
    if (ptr)
        ::free(ptr);
}

__attribute__((__weak__, __visibility__("default")))
void
operator delete(void* ptr, const std::nothrow_t&) throw ()
{
    ::operator delete(ptr);
}


__attribute__((__weak__, __visibility__("default")))
void
operator delete[] (void* ptr) throw ()
{
    ::operator delete (ptr);
}

__attribute__((__weak__, __visibility__("default")))
void
operator delete[] (void* ptr, const std::nothrow_t&) throw ()
{
    ::operator delete[](ptr);
}


namespace std
{

const nothrow_t nothrow = {};

new_handler
set_new_handler(new_handler handler) throw()
{
    new_handler r = __new_handler;
    __new_handler = handler;
    return r;
}

bad_alloc::bad_alloc() throw() 
{ 
}

bad_alloc::~bad_alloc() throw() 
{ 
}

const char* 
bad_alloc::what() const throw()
{
    return "std::bad_alloc";
}


bad_array_new_length::bad_array_new_length() throw()
{
}

bad_array_new_length::~bad_array_new_length() throw()
{
}

const char*
bad_array_new_length::what() const throw()
{
    return "bad_array_new_length";
}



void
__throw_bad_alloc()
{
    throw bad_alloc();
}

}  // std
