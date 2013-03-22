//===-- Metric.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Metric.h"

#include "CFCMutableArray.h"
#include "CFCMutableDictionary.h"
#include "CFCString.h"
#include "MemoryGauge.h"

using namespace lldb_perf;

template <class T>
Metric<T>::Metric () : Metric ("")
{
}

template <class T>
Metric<T>::Metric (const char* n, const char* d) :
    m_name(n ? n : ""),
    m_description(d ? d : ""),
    m_dataset ()
{
}

template <class T>
void
Metric<T>::Append (T v)
{
    m_dataset.push_back(v);
}

template <class T>
size_t
Metric<T>::GetCount () const
{
    return m_dataset.size();
}

template <class T>
T
Metric<T>::GetSum () const
{
    T sum = 0;
    for (auto v : m_dataset)
        sum += v;
    return sum;
}

template <class T>
T
Metric<T>::GetAverage () const
{
    return GetSum()/GetCount();
}

template <>
void Metric<double>::WriteImpl (CFCMutableDictionary& parent_dict, const char *name, const char *description, double value)
{
    assert(name && name[0]);
    CFCMutableDictionary dict;
    if (description && description[0])
        dict.AddValueCString(CFCString("description").get(),description, true);
    dict.AddValueDouble(CFCString("value").get(),value, true);
    parent_dict.AddValue(CFCString(name).get(), dict.get(), true);
}

template <>
void Metric<MemoryStats>::WriteImpl (CFCMutableDictionary& parent_dict, const char *name, const char *description, MemoryStats value)
{
    CFCMutableDictionary dict;
    if (description && description[0])
        dict.AddValueCString(CFCString("description").get(),description, true);
    CFCMutableDictionary value_dict;
    // don't write out the "virtual size", it doesn't mean anything useful as it includes
    // all of the shared cache and many other things that make it way too big to be useful
    //value_dict.AddValueUInt64(CFCString("virtual").get(), value.GetVirtualSize(), true);
    value_dict.AddValueUInt64(CFCString("resident").get(), value.GetResidentSize(), true);
    value_dict.AddValueUInt64(CFCString("max_resident").get(), value.GetMaxResidentSize(), true);
    
    dict.AddValue(CFCString("value").get(),value_dict.get(), true);

    parent_dict.AddValue(CFCString(name).get(), dict.get(), true);
}

template class lldb_perf::Metric<double>;
template class lldb_perf::Metric<MemoryStats>;
