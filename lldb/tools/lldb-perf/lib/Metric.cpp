//
//  Metric.cpp
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/7/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#include "Metric.h"

#include "CFCMutableArray.h"
#include "CFCMutableDictionary.h"
#include "CFCString.h"
#include "MemoryGauge.h"

using namespace lldb_perf;

template <class T>
Metric<T>::Metric () : Metric ("")
{}

template <class T>
Metric<T>::Metric (const char* n, const char* d) :
m_name(n ? n : ""),
m_description(d ? d : ""),
m_dataset ()
{}

template <class T>
void
Metric<T>::append (T v)
{
    m_dataset.push_back(v);
}

template <class T>
size_t
Metric<T>::count ()
{
    return m_dataset.size();
}

template <class T>
T
Metric<T>::sum ()
{
    T sum = 0;
    for (auto v : m_dataset)
        sum += v;
    return sum;
}

template <class T>
T
Metric<T>::average ()
{
    return sum()/count();
}

template <class T>
const char*
Metric<T>::name ()
{
    return m_name.c_str();
}

template <class T>
const char*
Metric<T>::description ()
{
    return m_description.c_str();
}

template <>
void Metric<double>::WriteImpl (CFCMutableArray& parent, identity<double>)
{
    CFCMutableDictionary dict;
    dict.AddValueCString(CFCString("name").get(),name(), true);
    dict.AddValueCString(CFCString("description").get(),description(), true);
    dict.AddValueDouble(CFCString("value").get(),this->average(), true);
    parent.AppendValue(dict.get(), true);
}

template <>
void Metric<MemoryStats>::WriteImpl (CFCMutableArray& parent, identity<MemoryStats>)
{
    CFCMutableDictionary dict;
    dict.AddValueCString(CFCString("name").get(),name(), true);
    dict.AddValueCString(CFCString("description").get(),description(), true);
    CFCMutableDictionary value;

    auto avg = this->average();
    
    value.AddValueUInt64(CFCString("virtual").get(), avg.GetVirtualSize(), true);
    value.AddValueUInt64(CFCString("resident").get(), avg.GetResidentSize(), true);
    value.AddValueUInt64(CFCString("max_resident").get(), avg.GetMaxResidentSize(), true);
    
    dict.AddValue(CFCString("value").get(),value.get(), true);
    
    parent.AppendValue(dict.get(), true);
}

template class lldb_perf::Metric<double>;
template class lldb_perf::Metric<MemoryStats>;
