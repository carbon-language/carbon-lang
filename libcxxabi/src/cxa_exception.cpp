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
#include "cxa_handlers.hpp"

namespace __cxxabiv1 {
static const uint64_t kOurExceptionClass          = 0x434C4E47432B2B00; // CLNGC++\0
static const uint64_t kOurDependentExceptionClass = 0x434C4E47432B2B01; // CLNGC++\1
                                                    
//  Utility routines
static inline __cxa_exception *exception_from_thrown_object(void *p) throw() {
    return ((__cxa_exception *) p) - 1;
}
    
static inline void * thrown_object_from_exception(void *p) throw() {
    return (void *) (((__cxa_exception *) p) + 1 );
}

static inline size_t object_size_from_exception_size(size_t size) throw() {
    return size + sizeof (__cxa_exception);
}

//  Get the exception object from the unwind pointer.
//  Relies on the structure layout, where the unwind pointer is right in
//  front of the user's exception object
static __cxa_exception *
exception_from_exception_object(void *ptr) throw() {
    _Unwind_Exception *p = reinterpret_cast<_Unwind_Exception *>(ptr);
    return exception_from_thrown_object(p + 1 );
}

static void setExceptionClass(_Unwind_Exception *unwind) throw() {
    unwind->exception_class = kOurExceptionClass;
}

static void setDependentExceptionClass(_Unwind_Exception *unwind) throw() {
    unwind->exception_class = kOurDependentExceptionClass;
}

//  Is it one of ours?
static bool isOurExceptionClass(_Unwind_Exception *unwind) throw() {
    return(unwind->exception_class == kOurExceptionClass)||
               (unwind->exception_class == kOurDependentExceptionClass);
}

static bool isDependentException(_Unwind_Exception *unwind) throw() {
    return (unwind->exception_class & 0xFF) == 0x01;
}

//  This does not need to be atomic
static inline int incrementHandlerCount(__cxa_exception *exception) throw() {
    return ++exception->handlerCount;
}

//  This does not need to be atomic
static inline  int decrementHandlerCount(__cxa_exception *exception) throw() {
    return --exception->handlerCount;
}

#include "fallback_malloc.cpp"

//  Allocate some memory from _somewhere_
static void *do_malloc(size_t size) throw() {
    void *ptr = std::malloc(size);
    if (NULL == ptr) // if malloc fails, fall back to emergency stash
        ptr = fallback_malloc(size);
    return ptr;
}

//  Didn't know you could "return <expression>" from a void function, did you?
//  Well, you can, if the type of the expression is "void" also.
static void do_free(void *ptr) throw() {
    return is_fallback_ptr(ptr) ? fallback_free(ptr) : std::free(ptr);
}

/*
    If reason isn't _URC_FOREIGN_EXCEPTION_CAUGHT, then the terminateHandler
    stored in exc is called.  Otherwise the exceptionDestructor stored in 
    exc is called, and then the memory for the exception is deallocated.
*/
static void exception_cleanup_func(_Unwind_Reason_Code reason, struct _Unwind_Exception* exc) {
    __cxa_exception *exception = exception_from_exception_object(exc);
    if (_URC_FOREIGN_EXCEPTION_CAUGHT != reason)
        std::__terminate(exception->terminateHandler);
        
    void * thrown_object = thrown_object_from_exception(exception);
    if (NULL != exception->exceptionDestructor)
        exception->exceptionDestructor(thrown_object);
    __cxa_free_exception(thrown_object);
}

static LIBCXXABI_NORETURN void failed_throw(__cxa_exception *exception) throw() {
//  Section 2.5.3 says:
//      * For purposes of this ABI, several things are considered exception handlers:
//      ** A terminate() call due to a throw.
//  and
//      * Upon entry, Following initialization of the catch parameter, 
//          a handler must call:
//      * void *__cxa_begin_catch(void *exceptionObject );
    (void) __cxa_begin_catch(&exception->unwindHeader);
    std::__terminate(exception->terminateHandler);
}

extern "C" {

//  Allocate a __cxa_exception object, and zero-fill it.
//  Reserve "thrown_size" bytes on the end for the user's exception
//  object. Zero-fill the object. If memory can't be allocated, call
//  std::terminate. Return a pointer to the memory to be used for the
//  user's exception object.
void * __cxa_allocate_exception (size_t thrown_size) throw() {
    size_t actual_size = object_size_from_exception_size(thrown_size);
    void *ptr = do_malloc(actual_size);
    if (NULL == ptr)
        std::terminate();
    std::memset(ptr, 0, actual_size);
    return thrown_object_from_exception(ptr);
}


//  Free a __cxa_exception object allocated with __cxa_allocate_exception.
void __cxa_free_exception (void * thrown_exception) throw() {
    do_free(exception_from_thrown_object(thrown_exception));
}


//  This function shall allocate a __cxa_dependent_exception and
//  return a pointer to it. (Really to the object, not past its' end).
//  Otherwise, it will work like __cxa_allocate_exception.
void * __cxa_allocate_dependent_exception () throw() {
    size_t actual_size = sizeof(__cxa_dependent_exception);
    void *ptr = do_malloc(actual_size);
    if (NULL == ptr)
        std::terminate();
    std::memset(ptr, 0, actual_size);
    return ptr;
}


//  This function shall free a dependent_exception.
//  It does not affect the reference count of the primary exception.
void __cxa_free_dependent_exception (void * dependent_exception) throw() {
    do_free(dependent_exception);
}


// 2.4.3 Throwing the Exception Object
/*
After constructing the exception object with the throw argument value,
the generated code calls the __cxa_throw runtime library routine. This
routine never returns.

The __cxa_throw routine will do the following:

* Obtain the __cxa_exception header from the thrown exception object address,
which can be computed as follows: 
 __cxa_exception *header = ((__cxa_exception *) thrown_exception - 1); 
* Save the current unexpected_handler and terminate_handler in the __cxa_exception header.
* Save the tinfo and dest arguments in the __cxa_exception header. 
* Set the exception_class field in the unwind header. This is a 64-bit value
representing the ASCII string "XXXXC++\0", where "XXXX" is a
vendor-dependent string. That is, for implementations conforming to this
ABI, the low-order 4 bytes of this 64-bit value will be "C++\0".
* Increment the uncaught_exception flag. 
* Call _Unwind_RaiseException in the system unwind library, Its argument is the
pointer to the thrown exception, which __cxa_throw itself received as an argument.
__Unwind_RaiseException begins the process of stack unwinding, described
in Section 2.5. In special cases, such as an inability to find a
handler, _Unwind_RaiseException may return. In that case, __cxa_throw
will call terminate, assuming that there was no handler for the
exception.
*/
LIBCXXABI_NORETURN void 
__cxa_throw(void * thrown_exception, std::type_info * tinfo, void (*dest)(void *)) {
    __cxa_eh_globals *globals = __cxa_get_globals();
    __cxa_exception *exception = exception_from_thrown_object(thrown_exception);

    exception->unexpectedHandler = std::get_unexpected();
    exception->terminateHandler  = std::get_terminate();
    exception->exceptionType = tinfo;
    exception->exceptionDestructor = dest;
    setExceptionClass(&exception->unwindHeader);
    exception->referenceCount = 1;  // This is a newly allocated exception, no need for thread safety.
    globals->uncaughtExceptions += 1;   // Not atomically, since globals are thread-local

    exception->unwindHeader.exception_cleanup = exception_cleanup_func;
#if __arm__
    _Unwind_SjLj_RaiseException(&exception->unwindHeader);
#else
    _Unwind_RaiseException(&exception->unwindHeader);
#endif
//  If we get here, some kind of unwinding error has occurred.
    failed_throw(exception);
}


// 2.5.3 Exception Handlers
extern void * __cxa_get_exception_ptr(void * exceptionObject) throw() {
//    TODO:  This must return the *adjusted* pointer to the exception object.
    return exception_from_exception_object(exceptionObject);
}
    

/*
This routine:
* Increment's the exception's handler count.
* Places the exception on the stack of currently-caught exceptions if it is not 
  already there, linking the exception to the previous top of the stack.
* Decrements the uncaught_exception count.
* Returns the adjusted pointer to the exception object.
*/
void * __cxa_begin_catch(void * exceptionObject) throw() {
    __cxa_eh_globals *globals = __cxa_get_globals();
    __cxa_exception *exception = exception_from_exception_object(exceptionObject);

// TODO:  Handle foreign exceptions?  How?

//  Increment the handler count, removing the flag about being rethrown
    exception->handlerCount = exception->handlerCount < 0 ?
        -exception->handlerCount + 1 : exception->handlerCount + 1;

//  place the exception on the top of the stack if it's not there.
    if (exception != globals->caughtExceptions) {
        exception->nextException = globals->caughtExceptions;
        globals->caughtExceptions = exception;
    }
        
    globals->uncaughtExceptions -= 1;   // Not atomically, since globals are thread-local
    return __cxa_get_exception_ptr(exceptionObject);
}


/*
Upon exit for any reason, a handler must call:
    void __cxa_end_catch ();

This routine:
* Locates the most recently caught exception and decrements its handler count.
* Removes the exception from the caught exception stack, if the handler count goes to zero.
* If the handler count goes down to zero, and the exception was not re-thrown
  by throw, it locates the primary exception (which may be the same as the one
  it's handling) and decrements its reference count. If that reference count
  goes to zero, the function destroys the exception. In any case, if the current
  exception is a dependent exception, it destroys that.
*/
void __cxa_end_catch() {
    __cxa_eh_globals *globals = __cxa_get_globals_fast(); // __cxa_get_globals called in __cxa_begin_catch
    __cxa_exception *current_exception = globals->caughtExceptions;
    
    if (NULL != current_exception) {
        // TODO:  Handle foreign exceptions?  How?
        if (current_exception->handlerCount < 0) {
        //  The exception has been rethrown
            if (0 == incrementHandlerCount(current_exception)) {
                //  Remove from the chain of uncaught exceptions
                globals->caughtExceptions = current_exception->nextException;
                // but don't destroy
            }
        }
        else {  // The exception has not been rethrown
            if (0 == decrementHandlerCount(current_exception)) {
                //  Remove from the chain of uncaught exceptions
                globals->caughtExceptions = current_exception->nextException;
                if (isDependentException(&current_exception->unwindHeader)) {
                    // Reset current_exception to primaryException and deallocate the dependent exception
                    __cxa_dependent_exception* deh =
                        reinterpret_cast<__cxa_dependent_exception*>(current_exception + 1) - 1;
                    current_exception = static_cast<__cxa_exception*>(deh->primaryException) - 1;
                    __cxa_free_dependent_exception(deh);
                }
                // Destroy the primary exception only if its referenceCount goes to 0
                //    (this decrement must be atomic)
                __cxa_decrement_exception_refcount(thrown_object_from_exception(current_exception));
            }
        }       
    }
}


std::type_info * __cxa_current_exception_type() {
// FIXME:  I don't think we need to drill down into __cxa_dependent_exception
//         because the exceptionType is replicated in the __cxa_dependent_exception
//  get the current exception
    __cxa_eh_globals *globals = __cxa_get_globals_fast();
    if (NULL == globals)
        return NULL;     //  If there have never been any exceptions, there are none now.
    __cxa_exception *current_exception = globals->caughtExceptions;
    if (NULL == current_exception)
        return NULL;        //  No current exception
    if (isDependentException(&current_exception->unwindHeader)) {
        __cxa_dependent_exception* deh =
            reinterpret_cast<__cxa_dependent_exception*>(current_exception + 1) - 1;
        current_exception = static_cast<__cxa_exception*>(deh->primaryException) - 1;
    }
    return current_exception->exceptionType;
}

// 2.5.4 Rethrowing Exceptions
/*  This routine 
* marks the exception object on top of the caughtExceptions stack 
  (in an implementation-defined way) as being rethrown. 
* If the caughtExceptions stack is empty, it calls terminate() 
  (see [C++FDIS] [except.throw], 15.1.8). 
* It then calls _Unwind_Resume_or_Rethrow which should not return
   (terminate if it does).
*/
extern LIBCXXABI_NORETURN void __cxa_rethrow() {
    __cxa_eh_globals *globals = __cxa_get_globals();
    __cxa_exception *exception = globals->caughtExceptions;

    if (NULL == exception)   // there's no current exception!
        std::terminate ();

// TODO:  Handle foreign exceptions?  How?

//  Mark the exception as being rethrown (reverse the effects of __cxa_begin_catch)
    exception->handlerCount = -exception->handlerCount;
    globals->uncaughtExceptions += 1;
//  __cxa_end_catch will remove this exception from the caughtExceptions stack if necessary
    
#if __arm__
    (void) _Unwind_SjLj_Resume_or_Rethrow(&exception->unwindHeader);
#else
    (void) _Unwind_Resume_or_Rethrow     (&exception->unwindHeader);
#endif

//  If we get here, some kind of unwinding error has occurred.
    failed_throw(exception);
}

/*
    If p is not null, atomically increment the referenceCount field of the
    __cxa_exception header associated with the thrown object referred to by p.
*/
void
__cxa_increment_exception_refcount(void* p) throw()
{
    if (p != NULL )
    {
        __cxa_exception* header = exception_from_thrown_object(p);
        __sync_add_and_fetch(&header->referenceCount, 1);
    }
}

/*
    If p is not null, atomically decrement the referenceCount field of the
    __cxa_exception header associated with the thrown object referred to by p.
    If the referenceCount drops to zero, destroy and deallocate the exception.
*/
void
__cxa_decrement_exception_refcount(void* thrown_object) throw()
{
    if (thrown_object != NULL )
    {
        __cxa_exception* header = exception_from_thrown_object(thrown_object);
        if (__sync_sub_and_fetch(&header->referenceCount, size_t(1)) == 0)
        {
            if (NULL != header->exceptionDestructor)
                header->exceptionDestructor(thrown_object);
            __cxa_free_exception(thrown_object);
        }
    }
}

/*
    Returns a pointer to the thrown object (if any) at the top of the
    caughtExceptions stack.  Atommically increment the exception's referenceCount.
    If there is no such thrown object, returns null.

    We can use __cxa_get_globals_fast here to get the globals because if there have
    been no exceptions thrown, ever, on this thread, we can return NULL without 
    the need to allocate the exception-handling globals.
*/
void*
__cxa_current_primary_exception() throw()
{
//  get the current exception
    __cxa_eh_globals* globals = __cxa_get_globals_fast();
    if (NULL == globals)
        return NULL;        //  If there are no globals, there is no exception
    __cxa_exception* current_exception = globals->caughtExceptions;
    if (NULL == current_exception)
        return NULL;        //  No current exception
    if (isDependentException(&current_exception->unwindHeader)) {
        __cxa_dependent_exception* deh =
            reinterpret_cast<__cxa_dependent_exception*>(current_exception + 1) - 1;
        current_exception = static_cast<__cxa_exception*>(deh->primaryException) - 1;
    }
    void* thrown_object = thrown_object_from_exception(current_exception);
    __cxa_increment_exception_refcount(thrown_object);
    return thrown_object;
}

/*
    If reason isn't _URC_FOREIGN_EXCEPTION_CAUGHT, then the terminateHandler
    stored in exc is called.  Otherwise the referenceCount stored in the
    primary exception is decremented, destroying the primary if necessary.
    Finally the dependent exception is destroyed.
*/
static
void
dependent_exception_cleanup(_Unwind_Reason_Code reason, struct _Unwind_Exception* exc)
{
    __cxa_dependent_exception* deh = 
                      reinterpret_cast<__cxa_dependent_exception*>(exc + 1) - 1;
    if (_URC_FOREIGN_EXCEPTION_CAUGHT != reason)
        std::__terminate(deh->terminateHandler);
    __cxa_decrement_exception_refcount(deh->primaryException);
    __cxa_free_dependent_exception(deh);
}

/*
    If thrown_object is not null, allocate, initialize and thow a dependent
    exception.
*/
void
__cxa_rethrow_primary_exception(void* thrown_object)
{
    if ( thrown_object != NULL )
    {
        __cxa_exception* header = exception_from_thrown_object(thrown_object);
        __cxa_dependent_exception* deh =
            (__cxa_dependent_exception*)__cxa_allocate_dependent_exception();
        deh->primaryException = thrown_object;
        __cxa_increment_exception_refcount(thrown_object);
        deh->exceptionType = header->exceptionType;
        deh->unexpectedHandler = std::get_unexpected();
        deh->terminateHandler = std::get_terminate();
        setDependentExceptionClass(&deh->unwindHeader);
        __cxa_get_globals()->uncaughtExceptions += 1;
        deh->unwindHeader.exception_cleanup = dependent_exception_cleanup;
#if __arm__
        _Unwind_SjLj_RaiseException(&deh->unwindHeader);
#else
        _Unwind_RaiseException(&deh->unwindHeader);
#endif
        // Some sort of unwinding error.  Note that terminate is a handler.
        __cxa_begin_catch(&deh->unwindHeader);
    }
    // If we return client will call terminate()
}

}  // extern "C"

}  // abi
