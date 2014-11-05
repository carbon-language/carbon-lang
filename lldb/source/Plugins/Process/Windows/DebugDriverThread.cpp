//===-- DebugDriverThread.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DebugDriverThread.h"
#include "DriverMessages.h"
#include "DriverMessageResults.h"
#include "DebugOneProcessThread.h"
#include "ProcessMessages.h"

#include "lldb/Core/Log.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Target/Process.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

DebugDriverThread *DebugDriverThread::m_instance = NULL;

DebugDriverThread::DebugDriverThread()
{
    m_driver_thread = ThreadLauncher::LaunchThread("lldb.plugin.process-windows.driver-thread", DriverThread, this, nullptr);
    m_shutdown_event = ::CreateEvent(NULL, TRUE, FALSE, NULL);
    m_driver_message_event = ::CreateEvent(NULL, FALSE, FALSE, NULL);
    ::CreatePipe(&m_driver_pipe_read, &m_driver_pipe_write, NULL, 1024);
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
    m_driver_thread.Join(nullptr);

    ::CloseHandle(m_shutdown_event);
    ::CloseHandle(m_driver_message_event);
    ::CloseHandle(m_driver_pipe_read);
    ::CloseHandle(m_driver_pipe_write);

    m_shutdown_event = nullptr;
    m_driver_message_event = nullptr;
    m_driver_pipe_read = nullptr;
    m_driver_pipe_write = nullptr;
}

DebugDriverThread &
DebugDriverThread::GetInstance()
{
    return *m_instance;
}

void
DebugDriverThread::PostDebugMessage(const DriverMessage *message)
{
    message->Retain();
    if (!::WriteFile(m_driver_pipe_write, &message, sizeof(message), NULL, NULL))
    {
        message->Release();
        return;
    }

    ::SetEvent(m_driver_message_event);
}

const DriverMessageResult *
DebugDriverThread::HandleDriverMessage(const DriverMessage *message)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

    switch (message->GetMessageType())
    {
        case DriverMessageType::eLaunchProcess:
        {
            const auto *launch_message = static_cast<const DriverLaunchProcessMessage *>(message);
            return HandleDriverMessage(launch_message);
        }
        default:
            if (log)
                log->Printf("DebugDriverThread received unknown message type %d.", message->GetMessageType());
            return nullptr;
    }
}

const DriverLaunchProcessMessageResult *
DebugDriverThread::HandleDriverMessage(const DriverLaunchProcessMessage *launch_message)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    const char *exe = launch_message->GetLaunchInfo().GetExecutableFile().GetPath().c_str();
    if (log)
        log->Printf("DebugDriverThread launching process '%s'.", exe);

    // Create a DebugOneProcessThread which will do the actual creation and enter a debug loop on
    // a background thread, only returning after the process has been created on the background
    // thread.
    DebugMapEntry map_entry;
    map_entry.m_delegate = launch_message->GetDebugDelegate();
    map_entry.m_slave.reset(new DebugOneProcessThread(m_driver_thread));
    const DriverLaunchProcessMessageResult *result = map_entry.m_slave->DebugLaunch(launch_message);
    if (result && result->GetError().Success())
    {
        if (log)
            log->Printf("DebugDriverThread launched process '%s' with PID %d.", exe, result->GetProcess().GetProcessId());
        m_debugged_processes.insert(std::make_pair(result->GetProcess().GetProcessId(), map_entry));
    }
    else
    {
        if (log)
            log->Printf("An error occured launching process '%s' -- %s.", exe, result->GetError().AsCString());
    }
    return result;
}

void
DebugDriverThread::OnProcessLaunched(const ProcessMessageCreateProcess &message)
{
}

void
DebugDriverThread::OnExitProcess(const ProcessMessageExitProcess &message)
{
    lldb::pid_t pid = message.GetProcess().GetProcessId();

    // We invoke the delegate on the DriverThread rather than on the DebugOneProcessThread
    // so that invoking delegates is thread-safe amongst each other.  e.g. Two delegate invocations
    // are guaranteed to happen from the same thread.  Additionally, this guarantees that the
    // driver thread has a chance to clean up after itself before notifying processes of the debug
    // events, guaranteeing that no races happen whereby a process tries to kick off a new action
    // as a result of some event, but the request for that new action gets picked up by the driver
    // thread before the driver thread gets notified of the state change.
    auto iter = m_debugged_processes.find(pid);
    if (iter != m_debugged_processes.end())
        iter->second.m_delegate->OnExitProcess(message);

    m_debugged_processes.erase(iter);
}

void
DebugDriverThread::OnDebuggerConnected(const ProcessMessageDebuggerConnected &message)
{
}

void
DebugDriverThread::OnDebugException(const ProcessMessageException &message)
{
}

void
DebugDriverThread::OnCreateThread(const ProcessMessageCreateThread &message)
{
}

void
DebugDriverThread::OnExitThread(const ProcessMessageExitThread &message)
{
}

void
DebugDriverThread::OnLoadDll(const ProcessMessageLoadDll &message)
{
}

void
DebugDriverThread::OnUnloadDll(const ProcessMessageUnloadDll &message)
{
}

void
DebugDriverThread::OnDebugString(const ProcessMessageDebugString &message)
{
}

void
DebugDriverThread::OnDebuggerError(const ProcessMessageDebuggerError &message)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

    lldb::pid_t pid = message.GetProcess().GetProcessId();
    auto iter = m_debugged_processes.find(pid);
    if (iter != m_debugged_processes.end())
        iter->second.m_delegate->OnDebuggerError(message);
    m_debugged_processes.erase(iter);

    if (log)
    {
        log->Printf("An error was encountered while debugging process %d.  Debugging has been terminated.  Error = s.", pid,
                    message.GetError().AsCString());
    }
}

bool
DebugDriverThread::ProcessDriverMessages()
{
    DWORD bytes_available = 0;
    if (!PeekNamedPipe(m_driver_pipe_read, NULL, 0, NULL, &bytes_available, NULL))
    {
        // There's some kind of error with the named pipe.  Fail out and stop monitoring.
        return false;
    }

    if (bytes_available <= 0)
    {
        // There's no data available, but the operation succeeded.
        return true;
    }

    int count = bytes_available / sizeof(DriverMessage *);
    std::vector<DriverMessage *> messages(count);
    if (!::ReadFile(m_driver_pipe_read, &messages[0], bytes_available, NULL, NULL))
        return false;

    for (DriverMessage *message : messages)
    {
        const DriverMessageResult *result = HandleDriverMessage(message);
        message->CompleteMessage(result);
        message->Release();
    }
    return true;
}

lldb::thread_result_t
DebugDriverThread::DriverThread(void *data)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf("ProcessWindows DebugDriverThread starting up.");

    DebugDriverThread *driver_thread = static_cast<DebugDriverThread *>(data);
    const int kDriverMessageEventIndex = 0;
    const int kShutdownEventIndex = 1;

    Error error;
    HANDLE events[kShutdownEventIndex + 1];
    events[kDriverMessageEventIndex] = driver_thread->m_driver_message_event;
    events[kShutdownEventIndex] = driver_thread->m_shutdown_event;

    while (true)
    {
        bool exit = false;
        // See if any new processes are ready for debug monitoring.
        DWORD result = WaitForMultipleObjectsEx(llvm::array_lengthof(events), events, FALSE, 1000, TRUE);
        switch (result)
        {
            case WAIT_OBJECT_0 + kDriverMessageEventIndex:
                // LLDB is telling us to do something.  Process pending messages in our queue.
                driver_thread->ProcessDriverMessages();
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
        log->Printf("ProcessWindows Debug driver thread exiting.  %s", error.AsCString());
    return 0;
}
