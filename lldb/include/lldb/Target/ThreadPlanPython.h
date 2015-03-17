//===-- ThreadPlanPython.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlan_Python_h_
#define liblldb_ThreadPlan_Python_h_

// C Includes
// C++ Includes
#include <string>
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/StructuredData.h"
#include "lldb/Core/UserID.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanTracer.h"
#include "lldb/Target/StopInfo.h"

namespace lldb_private {

//------------------------------------------------------------------
//  ThreadPlanPython:
//
//------------------------------------------------------------------

class ThreadPlanPython : public ThreadPlan
{
public:
    ThreadPlanPython (Thread &thread, const char *class_name);
    virtual ~ThreadPlanPython ();
    
    virtual void
    GetDescription (Stream *s,
                    lldb::DescriptionLevel level);

    virtual bool
    ValidatePlan (Stream *error);

    virtual bool
    ShouldStop (Event *event_ptr);

    virtual bool
    MischiefManaged ();

    virtual bool
    WillStop ();

    virtual bool
    StopOthers ();

    virtual void
    DidPush ();

protected:
    virtual bool
    DoPlanExplainsStop (Event *event_ptr);
    
    virtual lldb::StateType
    GetPlanRunState ();

private:
  std::string m_class_name;
  StructuredData::ObjectSP m_implementation_sp;

    DISALLOW_COPY_AND_ASSIGN(ThreadPlanPython);
};


} // namespace lldb_private

#endif  // liblldb_ThreadPlan_Python_h_
