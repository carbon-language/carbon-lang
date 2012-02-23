//===------------------------- cxa_handlers.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//
// This file implements the functionality associated with the terminate_handler,
//   unexpected_handler, and new_handler.
//===----------------------------------------------------------------------===//

#include <stdexcept>
#include <new>
#include <exception>
#include "abort_message.h"
#include "cxxabi.h"
#include "cxa_handlers.hpp"
#include "cxa_exception.hpp"
#include "private_typeinfo.h"

namespace std
{

static const char* cause = "uncaught";

static void default_terminate_handler()
{
    // If there might be an uncaught exception
    using namespace __cxxabiv1;
    __cxa_eh_globals* globals = __cxa_get_globals_fast();
    if (globals)
    {
        __cxa_exception* exception_header = globals->caughtExceptions;
        // If there is an uncaught exception
        if (exception_header)
        {
            _Unwind_Exception* unwind_exception =
                reinterpret_cast<_Unwind_Exception*>(exception_header + 1) - 1;
            bool native_exception =
                (unwind_exception->exception_class   & get_vendor_and_language) == 
                                 (kOurExceptionClass & get_vendor_and_language);
            if (native_exception)
            {
                void* thrown_object =
                    unwind_exception->exception_class == kOurDependentExceptionClass ?
                        ((__cxa_dependent_exception*)exception_header)->primaryException :
                        exception_header + 1;
                const __shim_type_info* thrown_type =
                    static_cast<const __shim_type_info*>(exception_header->exceptionType);
                // Try to get demangled name of thrown_type
                int status;
                char buf[1024];
                size_t len = sizeof(buf);
                const char* name = __cxa_demangle(thrown_type->name(), buf, &len, &status);
                if (status != 0)
                    name = thrown_type->name();
                // If the uncaught exception can be caught with std::exception&
                const __shim_type_info* catch_type =
                    static_cast<const __shim_type_info*>(&typeid(exception));
                if (catch_type->can_catch(thrown_type, thrown_object))
                {
                    // Include the what() message from the exception
                    const exception* e = static_cast<const exception*>(thrown_object);
                    abort_message("terminating with %s exception of type %s: %s",
                                  cause, name, e->what());
                }
                else
                    // Else just note that we're terminating with an exception
                    abort_message("terminating with %s exception of type %s",
                                   cause, name);
            }
            else
                // Else we're terminating with a foreign exception
                abort_message("terminating with %s foreign exception", cause);
        }
    }
    // Else just note that we're terminating
    abort_message("terminating");
}

static void default_unexpected_handler()
{
    cause = "unexpected";
    terminate();
}

terminate_handler  __cxa_terminate_handler = default_terminate_handler;
unexpected_handler __cxa_unexpected_handler = default_unexpected_handler;
new_handler __cxa_new_handler = 0;

unexpected_handler
set_unexpected(unexpected_handler func) _NOEXCEPT
{
    if (func == 0)
        func = default_unexpected_handler;
    return __sync_lock_test_and_set(&__cxa_unexpected_handler, func);
}

unexpected_handler
get_unexpected() _NOEXCEPT
{
    return __sync_fetch_and_add(&__cxa_unexpected_handler, (unexpected_handler)0);
}

__attribute__((visibility("hidden"), noreturn))
void
__unexpected(unexpected_handler func)
{
    func();
    // unexpected handler should not return
    abort_message("unexpected_handler unexpectedly returned");
}

__attribute__((noreturn))
void
unexpected()
{
    __unexpected(get_unexpected());
}

terminate_handler
set_terminate(terminate_handler func) _NOEXCEPT
{
    if (func == 0)
        func = default_terminate_handler;
    return __sync_lock_test_and_set(&__cxa_terminate_handler, func);
}

terminate_handler
get_terminate() _NOEXCEPT
{
    return __sync_fetch_and_add(&__cxa_terminate_handler, (terminate_handler)0);
}

__attribute__((visibility("hidden"), noreturn))
void
__terminate(terminate_handler func) _NOEXCEPT
{
#if __has_feature(cxx_exceptions)
    try
    {
#endif  // __has_feature(cxx_exceptions)
        func();
        // handler should not return
        abort_message("terminate_handler unexpectedly returned");
#if __has_feature(cxx_exceptions)
    }
    catch (...)
    {
        // handler should not throw exception
        abort_message("terminate_handler unexpectedly threw an exception");
    }
#endif  // #if __has_feature(cxx_exceptions)
}

__attribute__((noreturn))
void
terminate() _NOEXCEPT
{
    // If there might be an uncaught exception
    using namespace __cxxabiv1;
    __cxa_eh_globals* globals = __cxa_get_globals_fast();
    if (globals)
    {
        __cxa_exception* exception_header = globals->caughtExceptions;
        if (exception_header)
        {
            _Unwind_Exception* unwind_exception =
                reinterpret_cast<_Unwind_Exception*>(exception_header + 1) - 1;
            bool native_exception =
                (unwind_exception->exception_class & get_vendor_and_language) ==
                               (kOurExceptionClass & get_vendor_and_language);
            if (native_exception)
            {
                __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
                __terminate(exception_header->terminateHandler);
            }
        }
    }
    __terminate(get_terminate());
}

new_handler
set_new_handler(new_handler handler) _NOEXCEPT
{
    return __sync_lock_test_and_set(&__cxa_new_handler, handler);
}

new_handler
get_new_handler() _NOEXCEPT
{
    return __sync_fetch_and_add(&__cxa_new_handler, (new_handler)0);
}

}  // std
