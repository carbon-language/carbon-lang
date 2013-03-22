//===-- Metric.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __PerfTestDriver__Metric__
#define __PerfTestDriver__Metric__

#include <vector>
#include <string>
#include <mach/task_info.h>

#include "CFCMutableDictionary.h"

namespace lldb_perf {

class MemoryStats;

class WriteResults
{
public:
    virtual void
    Write (CFCMutableDictionary& parent) = 0;
    
    virtual
    ~WriteResults () {}
};

template <class ValueType>
class Metric : public WriteResults
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
    GetName () const
    {
        if (m_name.empty())
            return NULL;
        return m_name.c_str();
    }

    const char*
    GetDescription () const
    {
        if (m_description.empty())
            return NULL;
        return m_description.c_str();
    }
    
    virtual void
    Write (CFCMutableDictionary& parent)
    {
        WriteImpl(parent, GetName(), GetDescription(), GetAverage());
    }

    static void WriteImpl (CFCMutableDictionary& parent, const char *name, const char *description, double);
    static void WriteImpl (CFCMutableDictionary& parent, const char *name, const char *description, MemoryStats);

private:
    std::string m_name;
    std::string m_description;
    std::vector<ValueType> m_dataset;
};
}

#endif /* defined(__PerfTestDriver__Metric__) */
