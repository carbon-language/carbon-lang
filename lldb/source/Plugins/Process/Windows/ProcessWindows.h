//===-- ProcessWindows.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_ProcessWindows_H_
#define liblldb_Plugins_Process_Windows_ProcessWindows_H_

// C Includes

// C++ Includes
#include <map>
#include <queue>

// Other libraries and framework includes
#include "ForwardDecl.h"
#include "IDebugDelegate.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Target/Process.h"

class ProcessMonitor;

namespace lldb_private
{
class HostProcess;
}

class ProcessWindows : public lldb_private::Process, public lldb_private::IDebugDelegate
{
public:
    //------------------------------------------------------------------
    // Static functions.
    //------------------------------------------------------------------
    static lldb::ProcessSP
    CreateInstance(lldb_private::Target& target,
                   lldb_private::Listener &listener,
                   const lldb_private::FileSpec *);

    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    //------------------------------------------------------------------
    // Constructors and destructors
    //------------------------------------------------------------------
    ProcessWindows(lldb_private::Target& target,
                   lldb_private::Listener &listener);

    ~ProcessWindows();

    virtual lldb_private::Error
    DoDetach(bool keep_stopped);

    virtual bool
    DetachRequiresHalt() { return true; }

    virtual bool
    UpdateThreadList(lldb_private::ThreadList &old_thread_list, lldb_private::ThreadList &new_thread_list);

    virtual lldb_private::Error
    DoLaunch (lldb_private::Module *exe_module,
              lldb_private::ProcessLaunchInfo &launch_info);

    virtual lldb_private::Error
    DoResume ();

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual void
    GetPluginCommandHelp(const char *command, lldb_private::Stream *strm);

    virtual lldb_private::Error
    ExecutePluginCommand(lldb_private::Args &command,
                         lldb_private::Stream *strm);

    virtual lldb_private::Log *
    EnablePluginLogging(lldb_private::Stream *strm,
                        lldb_private::Args &command);


    virtual bool
    CanDebug(lldb_private::Target &target, bool plugin_specified_by_name);

    virtual lldb_private::Error
    DoDestroy ();

    virtual void
    RefreshStateAfterStop ();

    virtual bool
    IsAlive ();

    virtual size_t DoReadMemory(lldb::addr_t vm_addr, void *buf, size_t size, lldb_private::Error &error);

    // IDebugDelegate overrides.
    virtual void OnProcessLaunched(const lldb_private::ProcessMessageCreateProcess &message) override;
    virtual void OnExitProcess(const lldb_private::ProcessMessageExitProcess &message) override;
    virtual void OnDebuggerConnected(const lldb_private::ProcessMessageDebuggerConnected &message) override;
    virtual void OnDebugException(const lldb_private::ProcessMessageException &message) override;
    virtual void OnCreateThread(const lldb_private::ProcessMessageCreateThread &message) override;
    virtual void OnExitThread(const lldb_private::ProcessMessageExitThread &message) override;
    virtual void OnLoadDll(const lldb_private::ProcessMessageLoadDll &message) override;
    virtual void OnUnloadDll(const lldb_private::ProcessMessageUnloadDll &message) override;
    virtual void OnDebugString(const lldb_private::ProcessMessageDebugString &message) override;
    virtual void OnDebuggerError(const lldb_private::ProcessMessageDebuggerError &message) override;
};

#endif  // liblldb_Plugins_Process_Windows_ProcessWindows_H_
