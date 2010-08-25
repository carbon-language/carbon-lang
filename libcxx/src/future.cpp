//===------------------------- future.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "future"
#include "string"

_LIBCPP_BEGIN_NAMESPACE_STD

class _LIBCPP_HIDDEN __future_error_category
    : public __do_message
{
public:
    virtual const char* name() const;
    virtual string message(int ev) const;
};

const char*
__future_error_category::name() const
{
    return "future";
}

string
__future_error_category::message(int ev) const
{
    switch (ev)
    {
    case future_errc::broken_promise:
        return string("The associated promise has been destructed prior "
                      "to the associated state becoming ready.");
    case future_errc::future_already_retrieved:
        return string("The future has already been retrieved from "
                      "the promise or packaged_task.");
    case future_errc::promise_already_satisfied:
        return string("The state of the promise has already been set.");
    case future_errc::no_state:
        return string("Operation not permitted on an object without "
                      "an associated state.");
    }
    return string("unspecified future_errc value\n");
}


const error_category&
future_category()
{
    static __future_error_category __f;
    return __f;
}

future_error::future_error(error_code __ec)
    : logic_error(__ec.message()),
      __ec_(__ec)
{
}

_LIBCPP_END_NAMESPACE_STD
