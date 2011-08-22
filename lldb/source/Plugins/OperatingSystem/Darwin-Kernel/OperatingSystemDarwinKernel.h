//===-- OperatingSystemMacOSXKernel.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OperatingSystemMacOSXKernel_h_
#define liblldb_OperatingSystemMacOSXKernel_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Target/OperatingSystem.h"

class DynamicRegisterInfo;

class OperatingSystemMacOSXKernel : public lldb_private::OperatingSystem
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static lldb_private::OperatingSystem *
    CreateInstance (lldb_private::Process *process, bool force);
    
    static void
    Initialize();
    
    static void
    Terminate();
    
    static const char *
    GetPluginNameStatic();
    
    static const char *
    GetPluginDescriptionStatic();
    
    //------------------------------------------------------------------
    // Class Methods
    //------------------------------------------------------------------
    OperatingSystemMacOSXKernel (lldb_private::Process *process);
    
    virtual
    ~OperatingSystemMacOSXKernel ();
    
    //------------------------------------------------------------------
    // lldb_private::PluginInterface Methods
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
    //------------------------------------------------------------------
    // lldb_private::OperatingSystem Methods
    //------------------------------------------------------------------
    virtual uint32_t
    UpdateThreadList (lldb_private::ThreadList &old_thread_list, 
                      lldb_private::ThreadList &new_thread_list);
    
    virtual void
    ThreadWasSelected (lldb_private::Thread *thread);

    virtual lldb::RegisterContextSP
    CreateRegisterContextForThread (lldb_private::Thread *thread);

    virtual lldb::StopInfoSP
    CreateThreadStopReason (lldb_private::Thread *thread);

protected:
    
    lldb::ValueObjectSP
    GetThreadListValueObject ();
    
    DynamicRegisterInfo *
    GetDynamicRegisterInfo ();

    lldb::ValueObjectSP m_thread_list_valobj_sp;
    std::auto_ptr<DynamicRegisterInfo> m_register_info_ap;
    
};

#endif // #ifndef liblldb_OperatingSystemMacOSXKernel_h_