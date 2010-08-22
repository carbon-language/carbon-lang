//===------------------------- thread.cpp----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "thread"
#include "exception"
#include <sys/types.h>
#include <sys/sysctl.h>

_LIBCPP_BEGIN_NAMESPACE_STD

thread::~thread()
{
    if (__t_ != 0)
        terminate();
}

void
thread::join()
{
    int ec = pthread_join(__t_, 0);
#ifndef _LIBCPP_NO_EXCEPTIONS
    if (ec)
        throw system_error(error_code(ec, system_category()), "thread::join failed");
#endif  // _LIBCPP_NO_EXCEPTIONS
    __t_ = 0;
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
#ifndef _LIBCPP_NO_EXCEPTIONS
    if (ec)
        throw system_error(error_code(ec, system_category()), "thread::detach failed");
#endif  // _LIBCPP_NO_EXCEPTIONS
}

unsigned
thread::hardware_concurrency()
{
#if defined(CTL_HW) && defined(HW_NCPU)
    int n;
    int mib[2] = {CTL_HW, HW_NCPU};
    std::size_t s = sizeof(n);
    sysctl(mib, 2, &n, &s, 0, 0);
    return n;
#else  // defined(CTL_HW) && defined(HW_NCPU)
    // TODO: grovel through /proc or check cpuid on x86 and similar
    // instructions on other architectures.
    return 0;  // Means not computable [thread.thread.static]
#endif  // defined(CTL_HW) && defined(HW_NCPU)
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
