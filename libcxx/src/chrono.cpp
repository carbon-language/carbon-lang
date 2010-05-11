//===------------------------- chrono.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "chrono"
#include <sys/time.h>        //for gettimeofday and timeval
#include <mach/mach_time.h>  // mach_absolute_time, mach_timebase_info_data_t

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono
{

// system_clock

system_clock::time_point
system_clock::now()
{
    timeval tv;
    gettimeofday(&tv, 0);
    return time_point(seconds(tv.tv_sec) + microseconds(tv.tv_usec));
}

time_t
system_clock::to_time_t(const time_point& t)
{
    return time_t(duration_cast<seconds>(t.time_since_epoch()).count());
}

system_clock::time_point
system_clock::from_time_t(time_t t)
{
    return system_clock::time_point(seconds(t));
}

// monotonic_clock

//   mach_absolute_time() * MachInfo.numer / MachInfo.denom is the number of
//   nanoseconds since the computer booted up.  MachInfo.numer and MachInfo.denom
//   are run time constants supplied by the OS.  This clock has no relationship
//   to the Gregorian calendar.  It's main use is as a high resolution timer.

// MachInfo.numer / MachInfo.denom is often 1 on the latest equipment.  Specialize
//   for that case as an optimization.

#pragma GCC visibility push(hidden)

static
monotonic_clock::rep
monotonic_simplified()
{
    return mach_absolute_time();
}

static
double
compute_monotonic_factor()
{
    mach_timebase_info_data_t MachInfo;
    mach_timebase_info(&MachInfo);
    return static_cast<double>(MachInfo.numer) / MachInfo.denom;
}

static
monotonic_clock::rep
monotonic_full()
{
    static const double factor = compute_monotonic_factor();
    return static_cast<monotonic_clock::rep>(mach_absolute_time() * factor);
}

typedef monotonic_clock::rep (*FP)();

static
FP
init_monotonic_clock()
{
    mach_timebase_info_data_t MachInfo;
    mach_timebase_info(&MachInfo);
    if (MachInfo.numer == MachInfo.denom)
        return &monotonic_simplified;
    return &monotonic_full;
}

#pragma GCC visiblity pop

monotonic_clock::time_point
monotonic_clock::now()
{
    static FP fp = init_monotonic_clock();
    return time_point(duration(fp()));
}

}

_LIBCPP_END_NAMESPACE_STD
