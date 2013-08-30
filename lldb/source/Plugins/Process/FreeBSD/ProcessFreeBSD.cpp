//===-- ProcessFreeBSD.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Target.h"

#include "ProcessFreeBSD.h"
#include "ProcessPOSIXLog.h"
#include "Plugins/Process/Utility/InferiorCallPOSIX.h"
#include "ProcessMonitor.h"
#include "POSIXThread.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------------------
// Static functions.

lldb::ProcessSP
ProcessFreeBSD::CreateInstance(Target& target,
                               Listener &listener,
                               const FileSpec *crash_file_path)
{
    lldb::ProcessSP process_sp;
    if (crash_file_path == NULL)
        process_sp.reset(new ProcessFreeBSD (target, listener));
    return process_sp;
}

void
ProcessFreeBSD::Initialize()
{
    static bool g_initialized = false;

    if (!g_initialized)
    {
        PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                      GetPluginDescriptionStatic(),
                                      CreateInstance);
        Log::Callbacks log_callbacks = {
            ProcessPOSIXLog::DisableLog,
            ProcessPOSIXLog::EnableLog,
            ProcessPOSIXLog::ListLogCategories
        };

        Log::RegisterLogChannel (ProcessFreeBSD::GetPluginNameStatic(), log_callbacks);
        ProcessPOSIXLog::RegisterPluginName(GetPluginNameStatic());
        g_initialized = true;
    }
}

lldb_private::ConstString
ProcessFreeBSD::GetPluginNameStatic()
{
    static ConstString g_name("freebsd");
    return g_name;
}

const char *
ProcessFreeBSD::GetPluginDescriptionStatic()
{
    return "Process plugin for FreeBSD";
}

//------------------------------------------------------------------------------
// ProcessInterface protocol.

lldb_private::ConstString
ProcessFreeBSD::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessFreeBSD::GetPluginVersion()
{
    return 1;
}

void
ProcessFreeBSD::GetPluginCommandHelp(const char *command, Stream *strm)
{
}

Error
ProcessFreeBSD::ExecutePluginCommand(Args &command, Stream *strm)
{
    return Error(1, eErrorTypeGeneric);
}

Log *
ProcessFreeBSD::EnablePluginLogging(Stream *strm, Args &command)
{
    return NULL;
}

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessFreeBSD::ProcessFreeBSD(Target& target, Listener &listener)
    : ProcessPOSIX(target, listener)
{
}

void
ProcessFreeBSD::Terminate()
{
}

Error
ProcessFreeBSD::DoDetach(bool keep_stopped)
{
    Error error;
    if (keep_stopped)
    {
        error.SetErrorString("Detaching with keep_stopped true is not currently supported on FreeBSD.");
        return error;
    }

    error = m_monitor->Detach(GetID());

    if (error.Success())
        SetPrivateState(eStateDetached);

    return error;
}

bool
ProcessFreeBSD::UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_THREAD));
    if (log && log->GetMask().Test(POSIX_LOG_VERBOSE))
        log->Printf ("ProcessFreeBSD::%s() (pid = %" PRIu64 ")", __FUNCTION__, GetID());

    bool has_updated = false;
    const lldb::pid_t pid = GetID();
    // Update the process thread list with this new thread.
    // FIXME: We should be using tid, not pid.
    assert(m_monitor);
    ThreadSP thread_sp (old_thread_list.FindThreadByID (pid, false));
    if (!thread_sp) {
        ProcessSP me = this->shared_from_this();
        thread_sp.reset(new POSIXThread(*me, pid));
        has_updated = true;
    }

    if (log && log->GetMask().Test(POSIX_LOG_VERBOSE))
        log->Printf ("ProcessFreeBSD::%s() updated tid = %" PRIu64, __FUNCTION__, pid);

    new_thread_list.AddThread(thread_sp);

    return has_updated; // the list has been updated
}
