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

void
__assoc_sub_state::__on_zero_shared()
{
    delete this;
}

void
__assoc_sub_state::set_value()
{
    unique_lock<mutex> __lk(__mut_);
    if (__has_value())
        throw future_error(make_error_code(future_errc::promise_already_satisfied));
    __state_ |= __constructed | ready;
    __lk.unlock();
    __cv_.notify_all();
}

void
__assoc_sub_state::set_value_at_thread_exit()
{
    unique_lock<mutex> __lk(__mut_);
    if (__has_value())
        throw future_error(make_error_code(future_errc::promise_already_satisfied));
    __state_ |= __constructed;
    __thread_local_data->__make_ready_at_thread_exit(this);
    __lk.unlock();
}

void
__assoc_sub_state::set_exception(exception_ptr __p)
{
    unique_lock<mutex> __lk(__mut_);
    if (__has_value())
        throw future_error(make_error_code(future_errc::promise_already_satisfied));
    __exception_ = __p;
    __state_ |= ready;
    __lk.unlock();
    __cv_.notify_all();
}

void
__assoc_sub_state::set_exception_at_thread_exit(exception_ptr __p)
{
    unique_lock<mutex> __lk(__mut_);
    if (__has_value())
        throw future_error(make_error_code(future_errc::promise_already_satisfied));
    __exception_ = __p;
    __thread_local_data->__make_ready_at_thread_exit(this);
    __lk.unlock();
}

void
__assoc_sub_state::__make_ready()
{
    unique_lock<mutex> __lk(__mut_);
    __state_ |= ready;
    __lk.unlock();
    __cv_.notify_all();
}

void
__assoc_sub_state::copy()
{
    unique_lock<mutex> __lk(__mut_);
    while (!__is_ready())
        __cv_.wait(__lk);
    if (__exception_ != nullptr)
        rethrow_exception(__exception_);
}

void
__assoc_sub_state::wait() const
{
    unique_lock<mutex> __lk(__mut_);
    while (!__is_ready())
        __cv_.wait(__lk);
}

future<void>::future(__assoc_sub_state* __state)
    : __state_(__state)
{
    if (__state_->__has_future_attached())
        throw future_error(make_error_code(future_errc::future_already_retrieved));
    __state_->__add_shared();
}

future<void>::~future()
{
    if (__state_)
        __state_->__release_shared();
}

void
future<void>::get()
{
    __assoc_sub_state* __s = __state_;
    __state_ = nullptr;
    return __s->copy();
}

promise<void>::promise()
    : __state_(new __assoc_sub_state)
{
}

promise<void>::~promise()
{
    if (__state_)
    {
        if (!__state_->__has_value() && __state_->use_count() > 1)
            __state_->set_exception(make_exception_ptr(
                      future_error(make_error_code(future_errc::broken_promise))
                                                      ));
        __state_->__release_shared();
    }
}

future<void>
promise<void>::get_future()
{
    if (__state_ == nullptr)
        throw future_error(make_error_code(future_errc::no_state));
    return future<void>(__state_);
}

void
promise<void>::set_value()
{
    if (__state_ == nullptr)
        throw future_error(make_error_code(future_errc::no_state));
    __state_->set_value();
}

void
promise<void>::set_exception(exception_ptr __p)
{
    if (__state_ == nullptr)
        throw future_error(make_error_code(future_errc::no_state));
    __state_->set_exception(__p);
}

void
promise<void>::set_value_at_thread_exit()
{
    if (__state_ == nullptr)
        throw future_error(make_error_code(future_errc::no_state));
    __state_->set_value_at_thread_exit();
}

void
promise<void>::set_exception_at_thread_exit(exception_ptr __p)
{
    if (__state_ == nullptr)
        throw future_error(make_error_code(future_errc::no_state));
    __state_->set_exception_at_thread_exit(__p);
}

_LIBCPP_END_NAMESPACE_STD
