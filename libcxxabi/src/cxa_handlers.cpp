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

namespace std
{

static const char* cause = "uncaught";

static void default_terminate_handler()
{
    std::exception_ptr cp = std::current_exception();
    if (cp)
    {
        try
        {
            rethrow_exception(cp);
        }
        catch (const std::exception& e)
        {
            abort_message("terminating with %s exception: %s\n", cause, e.what());
        }
        catch (...)
        {
            abort_message("terminating with %s exception\n", cause);
        }
    }
    abort_message("terminating\n");
}

static void default_unexpected_handler()
{
    cause = "unexpected";
    terminate();
}

static terminate_handler  __terminate_handler = default_terminate_handler;
static unexpected_handler __unexpected_handler = default_unexpected_handler;
static new_handler __new_handler = 0;

unexpected_handler
set_unexpected(unexpected_handler func) _NOEXCEPT
{
    if (func == 0)
        func = default_unexpected_handler;
    return __sync_lock_test_and_set(&__unexpected_handler, func);
}

unexpected_handler
get_unexpected() _NOEXCEPT
{
    return __sync_fetch_and_add(&__unexpected_handler, (unexpected_handler)0);
}

_LIBCPP_HIDDEN
_ATTRIBUTE(noreturn)
void
__unexpected(unexpected_handler func)
{
    func();
    // unexpected handler should not return
    abort_message("unexpected_handler unexpectedly returned");
}

_ATTRIBUTE(noreturn)
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
    return __sync_lock_test_and_set(&__terminate_handler, func);
}

terminate_handler
get_terminate() _NOEXCEPT
{
    return __sync_fetch_and_add(&__terminate_handler, (terminate_handler)0);
}

_LIBCPP_HIDDEN
_ATTRIBUTE(noreturn)
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

_ATTRIBUTE(noreturn)
void
terminate() _NOEXCEPT
{
    __terminate(get_terminate());
}

new_handler
set_new_handler(new_handler handler) _NOEXCEPT
{
    return __sync_lock_test_and_set(&__new_handler, handler);
}

new_handler
get_new_handler() _NOEXCEPT
{
    return __sync_fetch_and_add(&__new_handler, (new_handler)0);
}

}  // std
