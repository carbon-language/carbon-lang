//===---------------------- shared_mutex.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "__config"
#ifndef _LIBCPP_HAS_NO_THREADS

#define _LIBCPP_BUILDING_SHARED_MUTEX
#include "shared_mutex"

_LIBCPP_BEGIN_NAMESPACE_STD

shared_timed_mutex::shared_timed_mutex()
    : __state_(0)
{
}

// Exclusive ownership

void
shared_timed_mutex::lock()
{
    unique_lock<mutex> lk(__mut_);
    while (__state_ & __write_entered_)
        __gate1_.wait(lk);
    __state_ |= __write_entered_;
    while (__state_ & __n_readers_)
        __gate2_.wait(lk);
}

bool
shared_timed_mutex::try_lock()
{
    unique_lock<mutex> lk(__mut_);
    if (__state_ == 0)
    {
        __state_ = __write_entered_;
        return true;
    }
    return false;
}

void
shared_timed_mutex::unlock()
{
    lock_guard<mutex> _(__mut_);
    __state_ = 0;
    __gate1_.notify_all();
}

// Shared ownership

void
shared_timed_mutex::lock_shared()
{
    unique_lock<mutex> lk(__mut_);
    while ((__state_ & __write_entered_) || (__state_ & __n_readers_) == __n_readers_)
        __gate1_.wait(lk);
    unsigned num_readers = (__state_ & __n_readers_) + 1;
    __state_ &= ~__n_readers_;
    __state_ |= num_readers;
}

bool
shared_timed_mutex::try_lock_shared()
{
    unique_lock<mutex> lk(__mut_);
    unsigned num_readers = __state_ & __n_readers_;
    if (!(__state_ & __write_entered_) && num_readers != __n_readers_)
    {
        ++num_readers;
        __state_ &= ~__n_readers_;
        __state_ |= num_readers;
        return true;
    }
    return false;
}

void
shared_timed_mutex::unlock_shared()
{
    lock_guard<mutex> _(__mut_);
    unsigned num_readers = (__state_ & __n_readers_) - 1;
    __state_ &= ~__n_readers_;
    __state_ |= num_readers;
    if (__state_ & __write_entered_)
    {
        if (num_readers == 0)
            __gate2_.notify_one();
    }
    else
    {
        if (num_readers == __n_readers_ - 1)
            __gate1_.notify_one();
    }
}


_LIBCPP_END_NAMESPACE_STD

#endif // !_LIBCPP_HAS_NO_THREADS
