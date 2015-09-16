//===-- OperatingSystemGo.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#ifndef _liblldb_OperatingSystemGo_h_
#define _liblldb_OperatingSystemGo_h_

#include <iostream>

#include "lldb/Target/OperatingSystem.h"

class DynamicRegisterInfo;

class OperatingSystemGo : public lldb_private::OperatingSystem
{
  public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static lldb_private::OperatingSystem *CreateInstance(lldb_private::Process *process, bool force);

    static void Initialize();

    static void DebuggerInitialize(lldb_private::Debugger &debugger);

    static void Terminate();

    static lldb_private::ConstString GetPluginNameStatic();

    static const char *GetPluginDescriptionStatic();

    //------------------------------------------------------------------
    // Class Methods
    //------------------------------------------------------------------
    OperatingSystemGo(lldb_private::Process *process);

    virtual ~OperatingSystemGo();

    //------------------------------------------------------------------
    // lldb_private::PluginInterface Methods
    //------------------------------------------------------------------
    virtual lldb_private::ConstString GetPluginName();

    virtual uint32_t GetPluginVersion();

    //------------------------------------------------------------------
    // lldb_private::OperatingSystem Methods
    //------------------------------------------------------------------
    virtual bool UpdateThreadList(lldb_private::ThreadList &old_thread_list, lldb_private::ThreadList &real_thread_list,
                                  lldb_private::ThreadList &new_thread_list);

    virtual void ThreadWasSelected(lldb_private::Thread *thread);

    virtual lldb::RegisterContextSP CreateRegisterContextForThread(lldb_private::Thread *thread,
                                                                   lldb::addr_t reg_data_addr);

    virtual lldb::StopInfoSP CreateThreadStopReason(lldb_private::Thread *thread);

    //------------------------------------------------------------------
    // Method for lazy creation of threads on demand
    //------------------------------------------------------------------
    virtual lldb::ThreadSP CreateThread(lldb::tid_t tid, lldb::addr_t context);

  private:
    struct Goroutine;

    static lldb::ValueObjectSP FindGlobal(lldb::TargetSP target, const char *name);

    static lldb::TypeSP FindType(lldb::TargetSP target_sp, const char *name);

    bool Init(lldb_private::ThreadList &threads);

    Goroutine CreateGoroutineAtIndex(uint64_t idx, lldb_private::Error &err);

    std::unique_ptr<DynamicRegisterInfo> m_reginfo;
    lldb::ValueObjectSP m_allg_sp;
    lldb::ValueObjectSP m_allglen_sp;
};

#endif // #ifndef liblldb_OperatingSystemGo_h_
