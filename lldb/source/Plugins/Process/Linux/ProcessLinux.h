//===-- ProcessLinux.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessLinux_H_
#define liblldb_ProcessLinux_H_

// C Includes

// C++ Includes
#include <queue>

// Other libraries and framework includes
#include "lldb/Target/Process.h"
#include "ProcessMessage.h"

class ProcessMonitor;

class ProcessLinux :
    public lldb_private::Process
{
public:
    //------------------------------------------------------------------
    // Static functions.
    //------------------------------------------------------------------
    static Process*
    CreateInstance(lldb_private::Target& target,
                   lldb_private::Listener &listener);

    static void
    Initialize();

    static void
    Terminate();

    static const char *
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    //------------------------------------------------------------------
    // Constructors and destructors
    //------------------------------------------------------------------
    ProcessLinux(lldb_private::Target& target,
                 lldb_private::Listener &listener);

    virtual
    ~ProcessLinux();

    //------------------------------------------------------------------
    // Process protocol.
    //------------------------------------------------------------------
    virtual bool
    CanDebug(lldb_private::Target &target);

    virtual lldb_private::Error
    DoAttachToProcessWithID(lldb::pid_t pid);

    virtual lldb_private::Error
    DoLaunch(lldb_private::Module *module,
             char const *argv[],
             char const *envp[],
             uint32_t launch_flags,
             const char *stdin_path,
             const char *stdout_path,
             const char *stderr_path);

    virtual void
    DidLaunch();

    virtual lldb_private::Error
    DoResume();

    virtual lldb_private::Error
    DoHalt(bool &caused_stop);

    virtual lldb_private::Error
    DoDetach();

    virtual lldb_private::Error
    DoSignal(int signal);

    virtual lldb_private::Error
    DoDestroy();

    virtual void
    RefreshStateAfterStop();

    virtual bool
    IsAlive();

    virtual size_t
    DoReadMemory(lldb::addr_t vm_addr,
                 void *buf,
                 size_t size,
                 lldb_private::Error &error);

    virtual size_t
    DoWriteMemory(lldb::addr_t vm_addr, const void *buf, size_t size,
                  lldb_private::Error &error);

    virtual lldb::addr_t
    DoAllocateMemory(size_t size, uint32_t permissions,
                     lldb_private::Error &error);

    lldb::addr_t
    AllocateMemory(size_t size, uint32_t permissions,
                   lldb_private::Error &error);

    virtual lldb_private::Error
    DoDeallocateMemory(lldb::addr_t ptr);

    virtual size_t
    GetSoftwareBreakpointTrapOpcode(lldb_private::BreakpointSite* bp_site);

    virtual lldb_private::Error
    EnableBreakpoint(lldb_private::BreakpointSite *bp_site);

    virtual lldb_private::Error
    DisableBreakpoint(lldb_private::BreakpointSite *bp_site);

    virtual uint32_t
    UpdateThreadListIfNeeded();

    virtual lldb::ByteOrder
    GetByteOrder() const;

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
    GetPluginCommandHelp(const char *command, lldb_private::Stream *strm);

    virtual lldb_private::Error
    ExecutePluginCommand(lldb_private::Args &command,
                         lldb_private::Stream *strm);

    virtual lldb_private::Log *
    EnablePluginLogging(lldb_private::Stream *strm,
                        lldb_private::Args &command);

    //--------------------------------------------------------------------------
    // ProcessLinux internal API.

    /// Registers the given message with this process.
    void SendMessage(const ProcessMessage &message);

    ProcessMonitor &GetMonitor() { return *m_monitor; }

private:
    /// Target byte order.
    lldb::ByteOrder m_byte_order;

    /// Process monitor;
    ProcessMonitor *m_monitor;

    /// The module we are executing.
    lldb_private::Module *m_module;

    /// Message queue notifying this instance of inferior process state changes.
    lldb_private::Mutex m_message_mutex;
    std::queue<ProcessMessage> m_message_queue;

    /// Updates the loaded sections provided by the executable.
    ///
    /// FIXME:  It would probably be better to delegate this task to the
    /// DynamicLoader plugin, when we have one.
    void UpdateLoadedSections();

    /// Returns true if the process has exited.
    bool HasExited();
};

#endif  // liblldb_MacOSXProcess_H_
