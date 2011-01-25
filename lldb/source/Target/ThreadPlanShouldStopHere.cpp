//===-- ThreadPlanShouldStopHere.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanShouldStopHere.h"

using namespace lldb;
using namespace lldb_private;

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

//----------------------------------------------------------------------
// ThreadPlanShouldStopHere constructor
//----------------------------------------------------------------------
ThreadPlanShouldStopHere::ThreadPlanShouldStopHere(ThreadPlan *owner, ThreadPlanShouldStopHereCallback callback, void *baton) :
    m_callback (callback),
    m_baton (baton),
    m_owner (owner),
    m_flags (ThreadPlanShouldStopHere::eNone)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ThreadPlanShouldStopHere::~ThreadPlanShouldStopHere()
{
}

void
ThreadPlanShouldStopHere::SetShouldStopHereCallback (ThreadPlanShouldStopHereCallback callback, void *baton)
{
    m_callback = callback;
    m_baton = baton;
}

ThreadPlan *
ThreadPlanShouldStopHere::InvokeShouldStopHereCallback ()
{
    if (m_callback)
        return m_callback (m_owner, m_flags, m_baton);
    else
        return NULL;
}
