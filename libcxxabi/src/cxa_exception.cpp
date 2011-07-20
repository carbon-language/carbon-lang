//===------------------------- cxa_exception.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//  
//  This file implements the "Exception Handling APIs"
//  http://www.codesourcery.com/public/cxx-abi/abi-eh.html
//  
//===----------------------------------------------------------------------===//

#include "cxxabi.h"

#include <exception>        // for std::terminate
#include <cstdlib>          // for malloc, free
#include <string>           // for memset
#include <pthread.h>

#include "cxa_exception.hpp"
#include "cxa_exception_storage.hpp"

namespace __cxxabiv1 {

//  Utility routines
static __cxa_exception *exception_from_object ( void *p ) {
    return ((__cxa_exception *) p ) - 1;
    }
    
void * object_from_exception ( void *p ) {
    return (void *) (((__cxa_exception *) p ) + 1 );
    }

static size_t object_size_from_exception_size ( size_t size ) {
    return size + sizeof (__cxa_exception);
    }

#include "fallback_malloc.cpp"

//  Allocate some memory from _somewhere_
static void *do_malloc ( size_t size ) throw () {
    void *ptr = std::malloc ( size );
    if ( NULL == ptr )  // if malloc fails, fall back to emergency stash
        ptr = fallback_malloc ( size );
    return ptr;
    }

//  Didn't know you could "return <expression>" from a void function, did you?
//  Well, you can, if the type of the expression is "void" also.
static void do_free ( void *ptr ) throw () {
    return is_fallback_ptr ( ptr ) ? fallback_free ( ptr ) : std::free ( ptr );
    }

// pthread_once_t __globals::flag_ = PTHREAD_ONCE_INIT;

extern "C" {

//  Allocate a __cxa_exception object, and zero-fill it.
//  Reserve "thrown_size" bytes on the end for the user's exception
//  object. Zero-fill the object. If memory can't be allocated, call
//  std::terminate. Return a pointer to the memory to be used for the
//  user's exception object.
void * __cxa_allocate_exception (size_t thrown_size) throw() {
    size_t actual_size = object_size_from_exception_size ( thrown_size );
    void *ptr = do_malloc ( actual_size );
    if ( NULL == ptr )
        std::terminate ();
    std::memset ( ptr, 0, actual_size );
    return object_from_exception ( ptr );
    }


//  Free a __cxa_exception object allocated with __cxa_allocate_exception.
void __cxa_free_exception (void * thrown_exception) throw() {
    do_free ( exception_from_object ( thrown_exception ));
    }


//  This function shall allocate a __cxa_dependent_exception and
//  return a pointer to it. (Really to the object, not past its' end).
//  Otherwise, it will work like __cxa_allocate_exception.
void * __cxa_allocate_dependent_exception () throw() {
    size_t actual_size = sizeof ( __cxa_dependent_exception );
    void *ptr = do_malloc ( actual_size );
    if ( NULL == ptr )
        std::terminate ();
    std::memset ( ptr, 0, actual_size );
//  bookkeeping here ?
    return ptr;
    }


//  This function shall free a dependent_exception.
//  It does not affect the reference count of the primary exception.
void __cxa_free_dependent_exception (void * dependent_exception) throw() {
//  I'm pretty sure there's no bookkeeping here
    do_free ( dependent_exception );
    }

}  // extern "C"

}  // abi
