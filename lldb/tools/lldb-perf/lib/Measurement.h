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

namespace lldb_perf
{
template <typename GaugeType, typename Action>
class Measurement : public WriteToPList
{
public:
    Measurement () {}
    
    Measurement (Action act, const char* name = NULL, const char* descr = NULL)  :
    m_action (act),
    m_metric (Metric<typename GaugeType::SizeType>(name,descr))
    {}
    
    template <typename GaugeType_Rhs, typename Action_Rhs>
    Measurement (const Measurement<GaugeType_Rhs, Action_Rhs>& rhs) :
    m_action(rhs.action()),
    m_metric(rhs.metric())
    {
    }

    template <typename... Args>
    void
    operator () (Args... args)
    {
        GaugeType gauge;
        m_metric.append (gauge.gauge(m_action,args...));
    }
    
    virtual const Metric<typename GaugeType::SizeType>&
    metric () const
    {
        return m_metric;
    }
    
    virtual const Action&
    action () const
    {
        return m_action;
    }
    
    virtual void
    Write (CFCMutableArray& parent)
    {
        m_metric.Write(parent);
    }

protected:
    Action m_action;
    Metric<typename GaugeType::SizeType> m_metric;
};
    
template <typename Action>
class TimeMeasurement : public Measurement<TimeGauge,Action>
{
public:
    TimeMeasurement () : Measurement<TimeGauge,Action> ()
    {}
    
    TimeMeasurement (Action act, const char* name = NULL, const char* descr = NULL) : Measurement<TimeGauge,Action> (act, name, descr)
    {}
    
    template <typename Action_Rhs>
    TimeMeasurement (const TimeMeasurement<Action_Rhs>& rhs) : Measurement<TimeGauge,Action>(rhs)
    {}
    
    template <typename GaugeType_Rhs, typename Action_Rhs>
    TimeMeasurement (const Measurement<GaugeType_Rhs, Action_Rhs>& rhs) : Measurement<GaugeType_Rhs,Action_Rhs>(rhs)
    {}
    
    template <typename... Args>
    void
    operator () (Args... args)
    {
        Measurement<TimeGauge,Action>::operator()(args...);
    }
};

}

#endif /* defined(__PerfTestDriver__Measurement__) */
