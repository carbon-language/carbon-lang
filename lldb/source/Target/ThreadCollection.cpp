//===-- ThreadCollection.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdlib.h>

#include <algorithm>

#include "lldb/Target/ThreadCollection.h"

using namespace lldb;
using namespace lldb_private;

ThreadCollection::ThreadCollection() :
    m_threads(),
    m_mutex()
{
}

ThreadCollection::ThreadCollection(collection threads) :
    m_threads(threads),
    m_mutex()
{
}

void
ThreadCollection::AddThread (const ThreadSP &thread_sp)
{
    Mutex::Locker locker(GetMutex());
    m_threads.push_back (thread_sp);
}

void
ThreadCollection::InsertThread (const lldb::ThreadSP &thread_sp, uint32_t idx)
{
    Mutex::Locker locker(GetMutex());
    if (idx < m_threads.size())
        m_threads.insert(m_threads.begin() + idx, thread_sp);
    else
        m_threads.push_back (thread_sp);
}

uint32_t
ThreadCollection::GetSize ()
{
    Mutex::Locker locker(GetMutex());
    return m_threads.size();
}

ThreadSP
ThreadCollection::GetThreadAtIndex (uint32_t idx)
{
    Mutex::Locker locker(GetMutex());
    ThreadSP thread_sp;
    if (idx < m_threads.size())
        thread_sp = m_threads[idx];
    return thread_sp;
}
