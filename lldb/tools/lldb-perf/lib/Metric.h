//
//  Metric.h
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/7/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#ifndef __PerfTestDriver__Metric__
#define __PerfTestDriver__Metric__

#include <vector>
#include <string>
#include <mach/task_info.h>

#include "CFCMutableArray.h"

namespace lldb_perf {

class MemoryStats;

class WriteToPList
{
public:
    virtual void
    Write (CFCMutableArray& parent) = 0;
    
    virtual
    ~WriteToPList () {}
};

template <class ValueType>
class Metric : public WriteToPList
{
public:
    Metric ();
    Metric (const char*, const char* = NULL);
    
    void
    Append (ValueType v);
    
    ValueType
    GetAverage () const;
    
    size_t
    GetCount () const;
    
    ValueType
    GetSum () const;
    
    const char*
    GetName ()
    {
        return m_name.c_str();
    }

    const char*
    GetDescription ()
    {
        return m_description.c_str();
    }
    
    virtual void
    Write (CFCMutableArray& parent)
    {
        WriteImpl(parent, identity<ValueType>());
    }
    
private:

    template<typename T>
    struct identity { typedef T type; };
    
    void WriteImpl (CFCMutableArray& parent, identity<double>);

    void WriteImpl (CFCMutableArray& parent, identity<MemoryStats>);
    
    std::string m_name;
    std::string m_description;
    std::vector<ValueType> m_dataset;
};
}

#endif /* defined(__PerfTestDriver__Metric__) */
