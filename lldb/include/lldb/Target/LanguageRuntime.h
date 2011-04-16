//===-- LanguageRuntime.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LanguageRuntime_h_
#define liblldb_LanguageRuntime_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/Value.h"
#include "lldb/Target/ExecutionContextScope.h"

namespace lldb_private {

class LanguageRuntime :
    public PluginInterface
{
public:
    virtual
    ~LanguageRuntime();
    
    static LanguageRuntime* 
    FindPlugin (Process *process, lldb::LanguageType language);
    
    virtual lldb::LanguageType
    GetLanguageType () const = 0;
    
    virtual bool
    GetObjectDescription (Stream &str, ValueObject &object) = 0;
    
    virtual bool
    GetObjectDescription (Stream &str, Value &value, ExecutionContextScope *exe_scope) = 0;
    
    virtual bool
    GetDynamicValue (ValueObject &in_value, lldb::TypeSP &type_sp, Address &address) = 0;
    
    // This should be a fast test to determine whether it is likely that this value would
    // have a dynamic type.
    virtual bool
    CouldHaveDynamicValue (ValueObject &in_value) = 0;

    virtual void
    SetExceptionBreakpoints ()
    {
    }
    
    virtual void
    ClearExceptionBreakpoints ()
    {
    }
    
    virtual bool
    ExceptionBreakpointsExplainStop (lldb::StopInfoSP stop_reason)
    {
        return false;
    }
protected:
    //------------------------------------------------------------------
    // Classes that inherit from LanguageRuntime can see and modify these
    //------------------------------------------------------------------
    LanguageRuntime(Process *process);
    Process *m_process;
private:
    DISALLOW_COPY_AND_ASSIGN (LanguageRuntime);
};

} // namespace lldb_private

#endif  // liblldb_LanguageRuntime_h_
