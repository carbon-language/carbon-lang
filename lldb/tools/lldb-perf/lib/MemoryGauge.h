//===-- MemoryGauge.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __PerfTestDriver__MemoryGauge__
#define __PerfTestDriver__MemoryGauge__

#include "Gauge.h"

#include <mach/task_info.h>

namespace lldb_perf
{
class MemoryStats
{
public:
    MemoryStats (mach_vm_size_t virtual_size = 0,
                 mach_vm_size_t resident_size = 0,
                 mach_vm_size_t max_resident_size = 0);
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
public:
    MemoryGauge ();
    
    virtual
    ~MemoryGauge ()
    {
    }
    
    void
    Start ();
    
    SizeType
    Stop ();
    
    SizeType
    GetValue ();

private:
    enum class State
    {
        eNeverUsed,
        eCounting,
        eStopped
    };
    
    SizeType
    Now ();
    
    SizeType m_start;
    State m_state;
    SizeType m_value;
    
};
}

#endif /* defined(__PerfTestDriver__MemoryGauge__) */
