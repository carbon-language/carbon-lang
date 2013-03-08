//
//  MemoryGauge.h
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/6/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#ifndef __PerfTestDriver__MemoryGauge__
#define __PerfTestDriver__MemoryGauge__

#include "Gauge.h"

#include <mach/task_info.h>

namespace lldb { namespace perf
{
class MemoryGauge : public Gauge<mach_vm_size_t>
{
private:
    enum class State
    {
        eMSNeverUsed,
        eMSCounting,
        eMSStopped
    };
    
    SizeType
    now ();
    
    SizeType m_start;
    State m_state;
    SizeType m_value;

public:
    MemoryGauge ();
    
    virtual
    ~MemoryGauge ()
    {}
    
    void
    start ();
    
    SizeType
    stop ();
    
    SizeType
    value ();
};
} }

#endif /* defined(__PerfTestDriver__MemoryGauge__) */
