//
//  Gauge.h
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/7/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#ifndef PerfTestDriver_Gauge_h
#define PerfTestDriver_Gauge_h

#include <functional>

namespace lldb_perf
{
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
    start () = 0;
    
    virtual SizeType
    stop () = 0;
    
    virtual  SizeType
    value () = 0;
    
    template <typename F, typename... Args>
    SizeType
    gauge (F f,Args... args)
    {
        start();
        f(args...);
        return stop();
    }

};
}

#endif
