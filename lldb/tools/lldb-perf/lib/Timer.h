//
//  Timer.h
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/6/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#ifndef __PerfTestDriver__Timer__
#define __PerfTestDriver__Timer__

#include "Gauge.h"

#include <chrono>

using namespace std::chrono;

namespace lldb { namespace perf
{
class TimeGauge : public Gauge<double>
{
private:
    enum class State
    {
        eTSNeverUsed,
        eTSCounting,
        eTSStopped
    };
    
    typedef high_resolution_clock::time_point HPTime;
    HPTime m_start;
    double m_value;
    State m_state;
    
    HPTime
    now ();
    
public:
    TimeGauge ();
    
    virtual
    ~TimeGauge ()
    {}
    
    void
    start ();
    
    double
    stop ();
    
    double
    value ();
};
} }

#endif /* defined(__PerfTestDriver__Timer__) */
