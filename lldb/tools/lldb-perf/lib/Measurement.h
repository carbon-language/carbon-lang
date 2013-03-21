//
//  Measurement.h
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/7/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#ifndef __PerfTestDriver__Measurement__
#define __PerfTestDriver__Measurement__

#include "Gauge.h"
#include "Timer.h"
#include "Metric.h"
#include "MemoryGauge.h"

namespace lldb_perf
{
template <typename GaugeType, typename Action>
class Measurement : public WriteToPList
{
public:
    Measurement () :
        m_gauge (),
        m_action (),
        m_metric ()
    {
    }
    
    Measurement (Action act, const char* name = NULL, const char* desc = NULL)  :
        m_gauge (),
        m_action (act),
        m_metric (Metric<typename GaugeType::SizeType>(name, desc))
    {
    }
    
    template <typename GaugeType_Rhs, typename Action_Rhs>
    Measurement (const Measurement<GaugeType_Rhs, Action_Rhs>& rhs) :
        m_gauge(rhs.GetGauge()),
        m_action(rhs.GetAction()),
        m_metric(rhs.GetMetric())
    {
    }

    template <typename... Args>
    void
    operator () (Args... args)
    {
        m_metric.Append (m_gauge.Measure(m_action, args...));
    }
    
    virtual const Action&
    GetAction () const
    {
        return m_action;
    }
    
    virtual const GaugeType&
    GetGauge () const
    {
        return m_gauge;
    }
    
    virtual const Metric<typename GaugeType::SizeType>&
    GetMetric () const
    {
        return m_metric;
    }
    
    void
    Start ()
    {
        m_gauge.Start();
    }
    
    typename GaugeType::SizeType
    Stop ()
    {
        auto value = m_gauge.Stop();
        m_metric.Append(value);
        return value;
    }
        
    virtual void
    Write (CFCMutableArray& parent)
    {
        m_metric.Write(parent);
    }

protected:
    GaugeType m_gauge;
    Action m_action;
    Metric<typename GaugeType::SizeType> m_metric;
};
    
template <typename Action>
class TimeMeasurement : public Measurement<TimeGauge,Action>
{
public:
    TimeMeasurement () :
        Measurement<TimeGauge,Action> ()
    {
    }
    
    TimeMeasurement (Action act,
                     const char* name = NULL,
                     const char* descr = NULL) :
        Measurement<TimeGauge,Action> (act, name, descr)
    {
    }
    
    template <typename Action_Rhs>
    TimeMeasurement (const TimeMeasurement<Action_Rhs>& rhs) :
        Measurement<TimeGauge,Action>(rhs)
    {
    }
    
    template <typename GaugeType_Rhs, typename Action_Rhs>
    TimeMeasurement (const Measurement<GaugeType_Rhs, Action_Rhs>& rhs) :
        Measurement<GaugeType_Rhs,Action_Rhs>(rhs)
    {
    }
    
    template <typename... Args>
    void
    operator () (Args... args)
    {
        Measurement<TimeGauge,Action>::operator()(args...);
    }
};

template <typename Action>
class MemoryMeasurement : public Measurement<MemoryGauge,Action>
{
public:
    MemoryMeasurement () : Measurement<MemoryGauge,Action> ()
    {
    }
    
    MemoryMeasurement (Action act, const char* name = NULL, const char* descr = NULL) : Measurement<MemoryGauge,Action> (act, name, descr)
    {
    }
    
    template <typename Action_Rhs>
    MemoryMeasurement (const MemoryMeasurement<Action_Rhs>& rhs) : Measurement<MemoryGauge,Action>(rhs)
    {
    }
    
    template <typename GaugeType_Rhs, typename Action_Rhs>
    MemoryMeasurement (const Measurement<GaugeType_Rhs, Action_Rhs>& rhs) : Measurement<GaugeType_Rhs,Action_Rhs>(rhs)
    {
    }
    
    template <typename... Args>
    void
    operator () (Args... args)
    {
        Measurement<MemoryGauge,Action>::operator()(args...);
    }
};
    
}

#endif /* defined(__PerfTestDriver__Measurement__) */
