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
#include "FreeBSDThread.h"

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
    : ProcessPOSIX(target, listener),
      m_resume_signo(0)
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

    DisableAllBreakpointSites();

    error = m_monitor->Detach(GetID());

    if (error.Success())
        SetPrivateState(eStateDetached);

    return error;
}

Error
ProcessFreeBSD::DoResume()
{
    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PROCESS));

    SetPrivateState(eStateRunning);

    Mutex::Locker lock(m_thread_list.GetMutex());
    bool do_step = false;

    for (tid_collection::const_iterator t_pos = m_run_tids.begin(), t_end = m_run_tids.end(); t_pos != t_end; ++t_pos)
    {
        m_monitor->ThreadSuspend(*t_pos, false);
    }
    for (tid_collection::const_iterator t_pos = m_step_tids.begin(), t_end = m_step_tids.end(); t_pos != t_end; ++t_pos)
    {
        m_monitor->ThreadSuspend(*t_pos, false);
        do_step = true;
    }
    for (tid_collection::const_iterator t_pos = m_suspend_tids.begin(), t_end = m_suspend_tids.end(); t_pos != t_end; ++t_pos)
    {
        m_monitor->ThreadSuspend(*t_pos, true);
        // XXX Cannot PT_CONTINUE properly with suspended threads.
        do_step = true;
    }

    if (log)
        log->Printf("process %lu resuming (%s)", GetID(), do_step ? "step" : "continue");
    if (do_step)
        m_monitor->SingleStep(GetID(), m_resume_signo);
    else
        m_monitor->Resume(GetID(), m_resume_signo);

    return Error();
}

bool
ProcessFreeBSD::UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PROCESS));
    if (log)
        log->Printf("ProcessFreeBSD::%s (pid = %" PRIu64 ")", __FUNCTION__, GetID());

    std::vector<lldb::pid_t> tds;
    if (!GetMonitor().GetCurrentThreadIDs(tds))
    {
        return false;
    }

    ThreadList old_thread_list_copy(old_thread_list);
    for (size_t i = 0; i < tds.size(); ++i)
    {
        tid_t tid = tds[i];
        ThreadSP thread_sp (old_thread_list_copy.RemoveThreadByID(tid, false));
        if (!thread_sp)
        {
            thread_sp.reset(new FreeBSDThread(*this, tid));
            if (log)
                log->Printf("ProcessFreeBSD::%s new tid = %" PRIu64, __FUNCTION__, tid);
        }
        else
        {
            if (log)
                log->Printf("ProcessFreeBSD::%s existing tid = %" PRIu64, __FUNCTION__, tid);
        }
        new_thread_list.AddThread(thread_sp);
    }
    for (size_t i = 0; i < old_thread_list_copy.GetSize(false); ++i)
    {
        ThreadSP old_thread_sp(old_thread_list_copy.GetThreadAtIndex(i, false));
        if (old_thread_sp)
        {
            if (log)
                log->Printf("ProcessFreeBSD::%s remove tid", __FUNCTION__);
        }
    }

    return true;
}

Error
ProcessFreeBSD::WillResume()
{
    m_resume_signo = 0;
    m_suspend_tids.clear();
    m_run_tids.clear();
    m_step_tids.clear();
    return ProcessPOSIX::WillResume();
}

void
ProcessFreeBSD::SendMessage(const ProcessMessage &message)
{
    Mutex::Locker lock(m_message_mutex);

    switch (message.GetKind())
    {
    case ProcessMessage::eInvalidMessage:
        return;

    case ProcessMessage::eAttachMessage:
        SetPrivateState(eStateStopped);
        return;

    case ProcessMessage::eLimboMessage:
    case ProcessMessage::eExitMessage:
        m_exit_status = message.GetExitStatus();
        SetExitStatus(m_exit_status, NULL);
        break;

    case ProcessMessage::eSignalMessage:
    case ProcessMessage::eSignalDeliveredMessage:
    case ProcessMessage::eBreakpointMessage:
    case ProcessMessage::eTraceMessage:
    case ProcessMessage::eWatchpointMessage:
    case ProcessMessage::eCrashMessage:
        SetPrivateState(eStateStopped);
        break;

    case ProcessMessage::eNewThreadMessage:
        assert(0 && "eNewThreadMessage unexpected on FreeBSD");
        break;

    case ProcessMessage::eExecMessage:
        SetPrivateState(eStateStopped);
        break;
    }

    m_message_queue.push(message);
}

