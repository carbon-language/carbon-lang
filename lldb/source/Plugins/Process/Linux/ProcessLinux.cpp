//===-- ProcessLinux.cpp ----------------------------------------*- C++ -*-===//
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

#include "ProcessLinux.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "Plugins/Process/Utility/InferiorCallPOSIX.h"
#include "Plugins/Process/Utility/LinuxSignals.h"
#include "ProcessMonitor.h"
#include "LinuxThread.h"

using namespace lldb;
using namespace lldb_private;

namespace
{
    UnixSignalsSP&
    GetStaticLinuxSignalsSP ()
    {
        static UnixSignalsSP s_unix_signals_sp (new process_linux::LinuxSignals ());
        return s_unix_signals_sp;
    }
}

//------------------------------------------------------------------------------
// Static functions.

ProcessSP
ProcessLinux::CreateInstance(Target &target, Listener &listener, const FileSpec *core_file)
{
    return ProcessSP(new ProcessLinux(target, listener, (FileSpec *)core_file));
}

void
ProcessLinux::Initialize()
{
    static bool g_initialized = false;

    if (!g_initialized)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                      GetPluginDescriptionStatic(),
                                      CreateInstance);

        Log::Callbacks log_callbacks = {
            ProcessPOSIXLog::DisableLog,
            ProcessPOSIXLog::EnableLog,
            ProcessPOSIXLog::ListLogCategories
        };
        
        Log::RegisterLogChannel (ProcessLinux::GetPluginNameStatic(), log_callbacks);
        ProcessPOSIXLog::RegisterPluginName(GetPluginNameStatic());
    }
}

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessLinux::ProcessLinux(Target& target, Listener &listener, FileSpec *core_file)
    : ProcessPOSIX(target, listener, GetStaticLinuxSignalsSP ()), m_core_file(core_file), m_stopping_threads(false)
{
#if 0
    // FIXME: Putting this code in the ctor and saving the byte order in a
    // member variable is a hack to avoid const qual issues in GetByteOrder.
    ObjectFile *obj_file = GetTarget().GetExecutableModule()->GetObjectFile();
    m_byte_order = obj_file->GetByteOrder();
#else
    // XXX: Will work only for local processes.
    m_byte_order = lldb::endian::InlHostByteOrder();
#endif
}

void
ProcessLinux::Terminate()
{
}

lldb_private::ConstString
ProcessLinux::GetPluginNameStatic()
{
    static ConstString g_name("linux");
    return g_name;
}

const char *
ProcessLinux::GetPluginDescriptionStatic()
{
    return "Process plugin for Linux";
}


bool
ProcessLinux::UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    new_thread_list = old_thread_list;
    return new_thread_list.GetSize(false) > 0;
}


//------------------------------------------------------------------------------
// ProcessInterface protocol.

lldb_private::ConstString
ProcessLinux::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessLinux::GetPluginVersion()
{
    return 1;
}

void
ProcessLinux::GetPluginCommandHelp(const char *command, Stream *strm)
{
}

Error
ProcessLinux::ExecutePluginCommand(Args &command, Stream *strm)
{
    return Error(1, eErrorTypeGeneric);
}

Log *
ProcessLinux::EnablePluginLogging(Stream *strm, Args &command)
{
    return NULL;
}

Error
ProcessLinux::DoDetach(bool keep_stopped)
{
    Error error;
    if (keep_stopped)
    {
        // FIXME: If you want to implement keep_stopped,
        // this would be the place to do it.
        error.SetErrorString("Detaching with keep_stopped true is not currently supported on Linux.");
        return error;
    }

    Mutex::Locker lock(m_thread_list.GetMutex());

    uint32_t thread_count = m_thread_list.GetSize(false);
    for (uint32_t i = 0; i < thread_count; ++i)
    {
        POSIXThread *thread = static_cast<POSIXThread*>(
            m_thread_list.GetThreadAtIndex(i, false).get());
        error = m_monitor->Detach(thread->GetID());
    }

    if (error.Success())
        SetPrivateState(eStateDetached);

    return error;
}


// ProcessPOSIX override
void
ProcessLinux::StopAllThreads(lldb::tid_t stop_tid)
{
    // If a breakpoint occurs while we're stopping threads, we'll get back
    // here, but we don't want to do it again.  Only the MonitorChildProcess
    // thread calls this function, so we don't need to protect this flag.
    if (m_stopping_threads)
      return;
    m_stopping_threads = true;

    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessLinux::%s() stopping all threads", __FUNCTION__);

    // Walk the thread list and stop the other threads.  The thread that caused
    // the stop should already be marked as stopped before we get here.
    Mutex::Locker thread_list_lock(m_thread_list.GetMutex());

    uint32_t thread_count = m_thread_list.GetSize(false);
    for (uint32_t i = 0; i < thread_count; ++i)
    {
        POSIXThread *thread = static_cast<POSIXThread*>(
            m_thread_list.GetThreadAtIndex(i, false).get());
        assert(thread);
        lldb::tid_t tid = thread->GetID();
        if (!StateIsStoppedState(thread->GetState(), false))
            m_monitor->StopThread(tid);
    }

    m_stopping_threads = false;

    if (log)
        log->Printf ("ProcessLinux::%s() finished", __FUNCTION__);
}

// ProcessPOSIX override
POSIXThread *
ProcessLinux::CreateNewPOSIXThread(lldb_private::Process &process, lldb::tid_t tid)
{
    return new LinuxThread(process, tid);
}

bool
ProcessLinux::CanDebug(Target &target, bool plugin_specified_by_name)
{
    if (plugin_specified_by_name)
        return true;

    /* If core file is specified then let elf-core plugin handle it */
    if (m_core_file)
        return false;

    // If we're using llgs for local debugging, we must not say that this process
    // is used for debugging.
    if (PlatformLinux::UseLlgsForLocalDebugging ())
        return false;

    return ProcessPOSIX::CanDebug(target, plugin_specified_by_name);
}

