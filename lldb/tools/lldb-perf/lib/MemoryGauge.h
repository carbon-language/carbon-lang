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

namespace lldb_perf
{
class MemoryStats
{
public:
    MemoryStats ();
    MemoryStats (mach_vm_size_t,mach_vm_size_t = 0, mach_vm_size_t = 0);
    MemoryStats (const MemoryStats& rhs);
    
    MemoryStats&
    operator = (const MemoryStats& rhs);

    MemoryStats&
    operator += (const MemoryStats& rhs);

    MemoryStats
    operator - (const MemoryStats& rhs);

    MemoryStats&
    operator / (size_t rhs);
    
    mach_vm_size_t
    GetVirtualSize ()
    {
        return m_virtual_size;
    }
    
    mach_vm_size_t
    GetResidentSize ()
    {
        return m_resident_size;
    }
    
    mach_vm_size_t
    GetMaxResidentSize ()
    {
        return m_max_resident_size;
    }
    
    void
    SetVirtualSize (mach_vm_size_t vs)
    {
        m_virtual_size = vs;
    }
    
    void
    SetResidentSize (mach_vm_size_t rs)
    {
        m_resident_size = rs;
    }
    
    void
    SetMaxResidentSize (mach_vm_size_t mrs)
    {
        m_max_resident_size = mrs;
    }
    
private:
    mach_vm_size_t m_virtual_size;
    mach_vm_size_t m_resident_size;
    mach_vm_size_t m_max_resident_size;
};

class MemoryGauge : public Gauge<MemoryStats>
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
}

#endif /* defined(__PerfTestDriver__MemoryGauge__) */
