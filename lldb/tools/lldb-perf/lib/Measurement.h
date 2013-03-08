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
#include "Metric.h"

namespace lldb { namespace perf
{
template <typename GaugeType, typename Action>
class Measurement : public WriteToPList
{
public:
    Measurement (Action act, const char* name = NULL)  :
    m_action (act),
    m_metric (Metric<typename GaugeType::SizeType>(name))
    {}

    template <typename... Args>
    void
    operator () (Args... args)
    {
        GaugeType gauge;
        m_metric.append (gauge.gauge(m_action,args...));
    }
    
    Metric<typename GaugeType::SizeType>
    metric ()
    {
        return m_metric;
    }
    
    virtual void
    Write (CFCMutableArray& parent)
    {
        m_metric.Write(parent);
    }

private:
    Action m_action;
    Metric<typename GaugeType::SizeType> m_metric;
};
} }

#endif /* defined(__PerfTestDriver__Measurement__) */
