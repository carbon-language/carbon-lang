//===-- Gauge.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PerfTestDriver_Gauge_h
#define PerfTestDriver_Gauge_h

#include <functional>

namespace lldb_perf {

template <class TASizeType>
class Gauge
{
public:
    typedef TASizeType SizeType;
public:
    Gauge ()
    {}
    
    virtual
    ~Gauge ()
    {}
    
    virtual void
    Start () = 0;
    
    virtual SizeType
    Stop () = 0;
    
    virtual  SizeType
    GetValue () = 0;
    
    template <typename F, typename... Args>
    SizeType
    Measure (F f,Args... args)
    {
        Start();
        f(args...);
        return Stop();
    }

};
}

#endif
