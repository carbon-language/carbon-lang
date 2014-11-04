//===-- DebugDriverThread.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DebugDriverThread.h"
#include "DebugMonitorMessages.h"
#include "DebugMonitorMessageResults.h"
#include "DebugOneProcessThread.h"
#include "SlaveMessages.h"

#include "lldb/Core/Log.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Target/Process.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

DebugDriverThread *DebugDriverThread::m_instance = NULL;

DebugDriverThread::DebugDriverThread()
{
    m_monitor_thread = ThreadLauncher::LaunchThread("lldb.plugin.process-windows.monitor-thread", MonitorThread, this, nullptr);
    m_shutdown_event = ::CreateEvent(NULL, TRUE, FALSE, NULL);
    m_monitor_event = ::CreateEvent(NULL, FALSE, FALSE, NULL);
    ::CreatePipe(&m_monitor_pipe_read, &m_monitor_pipe_write, NULL, 1024);
}

DebugDriverThread::~DebugDriverThread()
{
}

void
DebugDriverThread::Initialize()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_VERBOSE));
    if (log)
        log->Printf("DebugDriverThread::Initialize");

    m_instance = new DebugDriverThread();
}

void
DebugDriverThread::Teardown()
{
    m_instance->Shutdown();

    delete m_instance;
    m_instance = nullptr;
}

void
DebugDriverThread::Shutdown()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_VERBOSE));
    if (log)
        log->Printf("DebugDriverThread::Shutdown");

    if (!m_shutdown_event)
        return;
    ::SetEvent(m_shutdown_event);
    m_monitor_thread.Join(nullptr);

    ::CloseHandle(m_shutdown_event);
    ::CloseHandle(m_monitor_event);
    ::CloseHandle(m_monitor_pipe_read);
    ::CloseHandle(m_monitor_pipe_write);

    m_shutdown_event = nullptr;
    m_monitor_event = nullptr;
    m_monitor_pipe_read = nullptr;
    m_monitor_pipe_write = nullptr;
}

DebugDriverThread &
DebugDriverThread::GetInstance()
{
    return *m_instance;
}

void
DebugDriverThread::PostDebugMessage(const DebugMonitorMessage *message)
{
    message->Retain();
    if (!::WriteFile(m_monitor_pipe_write, &message, sizeof(message), NULL, NULL))
    {
        message->Release();
        return;
    }

    ::SetEvent(m_monitor_event);
}

const DebugMonitorMessageResult *
DebugDriverThread::HandleMonitorMessage(const DebugMonitorMessage *message)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

    switch (message->GetMessageType())
    {
        case MonitorMessageType::eLaunchProcess:
        {
            const auto *launch_message = static_cast<const LaunchProcessMessage *>(message);
            return HandleMonitorMessage(launch_message);
        }
        default:
            if (log)
                log->Printf("DebugDriverThread received unknown message type %d.", message->GetMessageType());
            return nullptr;
    }
}

const LaunchProcessMessageResult *
DebugDriverThread::HandleMonitorMessage(const LaunchProcessMessage *launch_message)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    const char *exe = launch_message->GetLaunchInfo().GetExecutableFile().GetPath().c_str();
    if (log)
        log->Printf("DebugDriverThread launching process '%s'.", exe);

    // Create a DebugOneProcessThread which will do the actual creation and enter a debug loop on
    // a background thread, only returning after the process has been created on the background
    // thread.
    std::shared_ptr<DebugOneProcessThread> slave(new DebugOneProcessThread(m_monitor_thread));
    const LaunchProcessMessageResult *result = slave->DebugLaunch(launch_message);
    if (result && result->GetError().Success())
    {
        if (log)
            log->Printf("DebugDriverThread launched process '%s' with PID %d.", exe, result->GetProcess().GetProcessId());
        m_debugged_processes.insert(std::make_pair(result->GetProcess().GetProcessId(), slave));
    }
    else
    {
        if (log)
            log->Printf("An error occured launching process '%s' -- %s.", exe, result->GetError().AsCString());
    }
    return result;
}

void
DebugDriverThread::HandleSlaveEvent(const SlaveMessageProcessExited &message)
{
    lldb::pid_t pid = message.GetProcess().GetProcessId();

    m_debugged_processes.erase(pid);

    Process::SetProcessExitStatus(nullptr, pid, true, 0, message.GetExitCode());
}

void
DebugDriverThread::HandleSlaveEvent(const SlaveMessageRipEvent &message)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

    lldb::pid_t pid = message.GetProcess().GetProcessId();
    m_debugged_processes.erase(pid);

    if (log)
    {
        log->Printf("An error was encountered while debugging process %d.  Debugging has been terminated.  Error = s.", pid,
                    message.GetError().AsCString());
    }
}

bool
DebugDriverThread::ProcessMonitorMessages()
{
    DWORD bytes_available = 0;
    if (!PeekNamedPipe(m_monitor_pipe_read, NULL, 0, NULL, &bytes_available, NULL))
    {
        // There's some kind of error with the named pipe.  Fail out and stop monitoring.
        return false;
    }

    if (bytes_available <= 0)
    {
        // There's no data available, but the operation succeeded.
        return true;
    }

    int count = bytes_available / sizeof(DebugMonitorMessage *);
    std::vector<DebugMonitorMessage *> messages(count);
    if (!::ReadFile(m_monitor_pipe_read, &messages[0], bytes_available, NULL, NULL))
        return false;

    for (DebugMonitorMessage *message : messages)
    {
        const DebugMonitorMessageResult *result = HandleMonitorMessage(message);
        message->CompleteMessage(result);
        message->Release();
    }
    return true;
}

lldb::thread_result_t
DebugDriverThread::MonitorThread(void *data)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf("ProcessWindows DebugDriverThread starting up.");

    DebugDriverThread *monitor_thread = static_cast<DebugDriverThread *>(data);
    const int kMonitorEventIndex = 0;
    const int kShutdownEventIndex = 1;

    Error error;
    HANDLE events[kShutdownEventIndex + 1];
    events[kMonitorEventIndex] = monitor_thread->m_monitor_event;
    events[kShutdownEventIndex] = monitor_thread->m_shutdown_event;

    while (true)
    {
        bool exit = false;
        // See if any new processes are ready for debug monitoring.
        DWORD result = WaitForMultipleObjectsEx(llvm::array_lengthof(events), events, FALSE, 1000, TRUE);
        switch (result)
        {
            case WAIT_OBJECT_0 + kMonitorEventIndex:
                // LLDB is telling us to do something.  Process pending messages in our queue.
                monitor_thread->ProcessMonitorMessages();
                break;
            case WAIT_OBJECT_0 + kShutdownEventIndex:
                error.SetErrorString("Shutdown event received.");
                exit = true;
                break;
            case WAIT_TIMEOUT:
            case WAIT_IO_COMPLETION:
                break;
            default:
                error.SetError(GetLastError(), eErrorTypeWin32);
                exit = true;
                break;
        }
        if (exit)
            break;
    }

    if (log)
        log->Printf("ProcessWindows Debug monitor thread exiting.  %s", error.AsCString());
    return 0;
}
