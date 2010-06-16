//===-- ThreadSpec.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;

ThreadSpec::ThreadSpec() :
    m_index (-1),
    m_tid (LLDB_INVALID_THREAD_ID),
    m_name(),
    m_queue_name ()
{
}

ThreadSpec::ThreadSpec (const ThreadSpec &rhs) :
    m_index(rhs.m_index),
    m_tid(rhs.m_tid),
    m_name(rhs.m_name),
    m_queue_name(rhs.m_queue_name)
{
}

const ThreadSpec &
ThreadSpec::operator=(const ThreadSpec &rhs)
{
    m_index = rhs.m_index;
    m_tid = rhs.m_tid;
    m_name = rhs.m_name;
    m_queue_name = rhs.m_queue_name;
    return *this;
}

const char *
ThreadSpec::GetName () const
{
    if (m_name.empty())
        return NULL;
    else
        return m_name.c_str();
}

const char *
ThreadSpec::GetQueueName () const
{
    if (m_queue_name.empty())
        return NULL;
    else
        return m_queue_name.c_str();
}
