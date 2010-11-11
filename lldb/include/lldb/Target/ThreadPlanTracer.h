//===-- ThreadPlanTracer.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanTracer_h_
#define liblldb_ThreadPlanTracer_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/Thread.h"

namespace lldb_private {

class ThreadPlanTracer
{
friend class ThreadPlan;

public:
    ThreadPlanTracer (Thread &thread, lldb::StreamSP &stream_sp);    
    ThreadPlanTracer (Thread &thread);
        
    virtual ~ThreadPlanTracer()
    {
    }
    
    bool
    EnableTracing(bool value)
    {
        bool old_value = m_enabled;
        m_enabled = value;
        return old_value;
    }
    
    bool
    TracingEnabled()
    {
        return m_enabled;
    }
    
    bool
    EnableSingleStep (bool value)
    {
        bool old_value = m_single_step;
        m_single_step = value;
        return old_value;
    }
    
    bool 
    SingleStepEnabled ()
    {
        return m_single_step;
    }

protected:
    Thread &m_thread;

    Stream *
    GetLogStream ();
    
    virtual void Log();
    
private:
    bool
    TracerExplainsStop ();
        
    bool m_single_step;
    bool m_enabled;
    lldb::StreamSP m_stream_sp;
};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanTracer_h_
