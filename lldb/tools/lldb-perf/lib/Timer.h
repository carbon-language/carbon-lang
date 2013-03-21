//===-- Timer.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __PerfTestDriver__Timer__
#define __PerfTestDriver__Timer__

#include "Gauge.h"

#include <chrono>

using namespace std::chrono;

namespace lldb_perf
{
class TimeGauge : public Gauge<double>
{
private:
    enum class State
    {
        eNeverUsed,
        eCounting,
        eStopped
    };
    
    typedef high_resolution_clock::time_point TimeType;
    TimeType m_start;
    double m_value;
    State m_state;
    
    TimeType
    Now ();
    
public:
    TimeGauge ();
    
    virtual
    ~TimeGauge ()
    {
    }
    
    void
    Start ();
    
    double
    Stop ();
    
    double
    GetValue ();
};
}

#endif /* defined(__PerfTestDriver__Timer__) */
