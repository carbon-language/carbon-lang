//===-- ThreadPlan.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlan.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/State.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

ThreadPlanTracer::ThreadPlanTracer (Thread &thread, lldb::StreamSP &stream_sp) :
    m_single_step(true),
    m_enabled (false),
    m_thread (thread),
    m_stream_sp (stream_sp)
{
}

ThreadPlanTracer::ThreadPlanTracer (Thread &thread) :
    m_single_step(true),
    m_enabled (false),
    m_thread (thread),
    m_stream_sp ()
{
}

Stream *
ThreadPlanTracer::GetLogStream ()
{
    
    if (m_stream_sp.get())
        return m_stream_sp.get();
    else
        return &(m_thread.GetProcess().GetTarget().GetDebugger().GetOutputStream());
}

void 
ThreadPlanTracer::Log()
{
    SymbolContext sc;
    bool show_frame_index = false;
    bool show_fullpaths = false;
    
    m_thread.GetStackFrameAtIndex(0)->Dump (GetLogStream(), show_frame_index, show_fullpaths);
    GetLogStream()->Printf("\n");
}

bool
ThreadPlanTracer::TracerExplainsStop ()
{
    if (m_enabled && m_single_step)
    {
        lldb::StopInfoSP stop_info = m_thread.GetStopInfo();
        if (stop_info->GetStopReason() == eStopReasonTrace)
            return true;
        else 
            return false;
    }
    else
        return false;
}
