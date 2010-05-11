//===------------------------- thread.cpp----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "thread"
#include "exception"
#include <sys/sysctl.h>

_LIBCPP_BEGIN_NAMESPACE_STD

thread::~thread()
{
    if (__t_ != nullptr)
        terminate();
}

void
thread::join()
{
    int ec = pthread_join(__t_, 0);
    if (ec)
        throw system_error(error_code(ec, system_category()), "thread::join failed");
    __t_ = nullptr;
}

void
thread::detach()
{
    int ec = EINVAL;
    if (__t_ != 0)
    {
        ec = pthread_detach(__t_);
        if (ec == 0)
            __t_ = 0;
    }
    if (ec)
        throw system_error(error_code(ec, system_category()), "thread::detach failed");
}

unsigned
thread::hardware_concurrency()
{
    int n;
    int mib[2] = {CTL_HW, HW_NCPU};
    std::size_t s = sizeof(n);
    sysctl(mib, 2, &n, &s, 0, 0);
    return n;
}

namespace this_thread
{

void
sleep_for(const chrono::nanoseconds& ns)
{
    using namespace chrono;
    if (ns >= nanoseconds::zero())
    {
        timespec ts;
        ts.tv_sec = static_cast<decltype(ts.tv_sec)>(duration_cast<seconds>(ns).count());
        ts.tv_nsec = static_cast<decltype(ts.tv_nsec)>((ns - seconds(ts.tv_sec)).count());
        nanosleep(&ts, 0);
    }
}

}  // this_thread

_LIBCPP_END_NAMESPACE_STD
