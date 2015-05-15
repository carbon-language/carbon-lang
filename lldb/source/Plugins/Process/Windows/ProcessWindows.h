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
#include <memory>
#include <queue>

// Other libraries and framework includes
#include "ForwardDecl.h"
#include "IDebugDelegate.h"
#include "lldb/lldb-forward.h"
#include "lldb/Core/Error.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Target/Process.h"

#include "llvm/Support/Mutex.h"

class ProcessMonitor;

namespace lldb_private
{
class ProcessWindowsData;
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

    // lldb_private::Process overrides
    lldb_private::ConstString GetPluginName() override;
    uint32_t GetPluginVersion() override;

    size_t GetSTDOUT(char *buf, size_t buf_size, lldb_private::Error &error) override;
    size_t GetSTDERR(char *buf, size_t buf_size, lldb_private::Error &error) override;
    size_t PutSTDIN(const char *buf, size_t buf_size, lldb_private::Error &error) override;

    lldb_private::Error EnableBreakpointSite(lldb_private::BreakpointSite *bp_site) override;
    lldb_private::Error DisableBreakpointSite(lldb_private::BreakpointSite *bp_site) override;

    lldb_private::Error DoDetach(bool keep_stopped) override;
    lldb_private::Error DoLaunch(lldb_private::Module *exe_module, lldb_private::ProcessLaunchInfo &launch_info) override;
    lldb_private::Error DoResume() override;
    lldb_private::Error DoDestroy() override;
    lldb_private::Error DoHalt(bool &caused_stop) override;

    void DidLaunch() override;

    void RefreshStateAfterStop() override;
    lldb::addr_t GetImageInfoAddress() override;

    bool CanDebug(lldb_private::Target &target, bool plugin_specified_by_name) override;
    bool
    DetachRequiresHalt() override
    {
        return true;
    }
    bool
    DestroyRequiresHalt() override
    {
        return false;
    }
    bool UpdateThreadList(lldb_private::ThreadList &old_thread_list, lldb_private::ThreadList &new_thread_list) override;
    bool IsAlive() override;

    size_t DoReadMemory(lldb::addr_t vm_addr, void *buf, size_t size, lldb_private::Error &error) override;
    size_t DoWriteMemory(lldb::addr_t vm_addr, const void *buf, size_t size, lldb_private::Error &error) override;
    lldb_private::Error GetMemoryRegionInfo(lldb::addr_t vm_addr, lldb_private::MemoryRegionInfo &info) override;

    // IDebugDelegate overrides.
    void OnExitProcess(uint32_t exit_code) override;
    void OnDebuggerConnected(lldb::addr_t image_base) override;
    ExceptionResult OnDebugException(bool first_chance, const lldb_private::ExceptionRecord &record) override;
    void OnCreateThread(const lldb_private::HostThread &thread) override;
    void OnExitThread(const lldb_private::HostThread &thread) override;
    void OnLoadDll(const lldb_private::ModuleSpec &module_spec, lldb::addr_t module_addr) override;
    void OnUnloadDll(lldb::addr_t module_addr) override;
    void OnDebugString(const std::string &string) override;
    void OnDebuggerError(const lldb_private::Error &error, uint32_t type) override;

  private:
    llvm::sys::Mutex m_mutex;

    // Data for the active debugging session.
    std::unique_ptr<lldb_private::ProcessWindowsData> m_session_data;
};

#endif  // liblldb_Plugins_Process_Windows_ProcessWindows_H_
