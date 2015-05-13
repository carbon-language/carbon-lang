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
#include "Plugins/Process/POSIX/ProcessMessage.h"
#include "Plugins/Process/POSIX/ProcessPOSIX.h"

class ProcessMonitor;

namespace lldb_private {
namespace process_linux {

class ProcessLinux : public ProcessPOSIX
{
public:
    //------------------------------------------------------------------
    // Static functions.
    //------------------------------------------------------------------
    static lldb::ProcessSP
    CreateInstance(Target& target,
                   Listener &listener,
                   const FileSpec *);

    static void
    Initialize();

    static void
    Terminate();

    static ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    //------------------------------------------------------------------
    // Constructors and destructors
    //------------------------------------------------------------------
    ProcessLinux(Target& target,
                 Listener &listener,
                 const FileSpec *core_file);

    Error
    DoDetach(bool keep_stopped) override;

    bool
    DetachRequiresHalt() override { return true; }

    bool
    UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list) override;

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    ConstString
    GetPluginName() override;

    uint32_t
    GetPluginVersion() override;

    virtual void
    GetPluginCommandHelp(const char *command, Stream *strm);

    virtual Error
    ExecutePluginCommand(Args &command,
                         Stream *strm);

    virtual Log *
    EnablePluginLogging(Stream *strm,
                        Args &command);

    bool
    CanDebug(Target &target, bool plugin_specified_by_name) override;

    //------------------------------------------------------------------
    // ProcessPOSIX overrides
    //------------------------------------------------------------------
    void
    StopAllThreads(lldb::tid_t stop_tid) override;

    POSIXThread *
    CreateNewPOSIXThread(Process &process, lldb::tid_t tid) override;

private:

    const FileSpec *m_core_file;

    // Flag to avoid recursion when stopping all threads.
    bool m_stopping_threads;
};

} // namespace process_linux
} // namespace lldb_private

#endif  // liblldb_ProcessLinux_H_
