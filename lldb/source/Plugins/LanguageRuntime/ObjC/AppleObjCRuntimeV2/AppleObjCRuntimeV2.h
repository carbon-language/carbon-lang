//===-- AppleObjCRuntimeV2.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AppleObjCRuntimeV2_h_
#define liblldb_AppleObjCRuntimeV2_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Core/ValueObject.h"
#include "AppleObjCTrampolineHandler.h"
#include "AppleThreadPlanStepThroughObjCTrampoline.h"

namespace lldb_private {
    
class AppleObjCRuntimeV2 :
        public lldb_private::ObjCLanguageRuntime
{
public:
    ~AppleObjCRuntimeV2() { }
    
    // These are generic runtime functions:
    virtual bool
    GetObjectDescription (Stream &str, Value &value, ExecutionContextScope *exe_scope);
    
    virtual bool
    GetObjectDescription (Stream &str, ValueObject &object, ExecutionContextScope *exe_scope);
    
    virtual lldb::ValueObjectSP
    GetDynamicValue (lldb::ValueObjectSP in_value, ExecutionContextScope *exe_scope);

    // These are the ObjC specific functions.
    virtual bool
    IsModuleObjCLibrary (const lldb::ModuleSP &module_sp);
    
    virtual bool
    ReadObjCLibrary (const lldb::ModuleSP &module_sp);

    virtual bool
    HasReadObjCLibrary ()
    {
        return m_read_objc_library;
    }
    
    virtual lldb::ThreadPlanSP
    GetStepThroughTrampolinePlan (Thread &thread, bool stop_others);
    
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();
    
    static void
    Terminate();
    
    static lldb_private::LanguageRuntime *
    CreateInstance (Process *process, lldb::LanguageType language);
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
    virtual void
    GetPluginCommandHelp (const char *command, lldb_private::Stream *strm);
    
    virtual lldb_private::Error
    ExecutePluginCommand (lldb_private::Args &command, lldb_private::Stream *strm);
    
    virtual lldb_private::Log *
    EnablePluginLogging (lldb_private::Stream *strm, lldb_private::Args &command);
protected:
    Address *
    GetPrintForDebuggerAddr();
    
private:
    std::auto_ptr<Address>  m_PrintForDebugger_addr;
    bool m_read_objc_library;
    std::auto_ptr<lldb_private::AppleObjCTrampolineHandler> m_objc_trampoline_handler_ap;

    AppleObjCRuntimeV2(Process *process) : 
        lldb_private::ObjCLanguageRuntime(process),
        m_read_objc_library (false),
        m_objc_trampoline_handler_ap(NULL)
     { } // Call CreateInstance instead.
};
    
} // namespace lldb_private

#endif  // liblldb_AppleObjCRuntimeV2_h_
