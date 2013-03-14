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

namespace lldb { namespace perf
{
class WriteToPList
{
public:
    virtual void
    Write (CFCMutableArray& parent) = 0;
    
    virtual
    ~WriteToPList () {}
};

template <class ValueType>
class Metric : public WriteToPList {
public:
    Metric ();
    Metric (const char*, const char* = NULL);
    
    void
    append (ValueType v);
    
    size_t
    count ();
    
    ValueType
    sum ();
    
    ValueType
    average ();
    
    const char*
    name ();
    
    const char*
    description ();
    
    virtual void
    Write (CFCMutableArray& parent)
    {
        WriteImpl(parent, identity<ValueType>());
    }
    
private:

    template<typename T>
    struct identity { typedef T type; };
    
    void WriteImpl (CFCMutableArray& parent, identity<double>);

    void WriteImpl (CFCMutableArray& parent, identity<mach_vm_size_t>);
    
    std::string m_name;
    std::string m_description;
    std::vector<ValueType> m_dataset;
};
} }

#endif /* defined(__PerfTestDriver__Metric__) */
