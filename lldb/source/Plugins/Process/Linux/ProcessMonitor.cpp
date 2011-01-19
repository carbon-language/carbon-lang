//===-- ProcessMonitor.cpp ------------------------------------ -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>
#include <poll.h>
#include <string.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/PseudoTerminal.h"

#include "LinuxThread.h"
#include "ProcessLinux.h"
#include "ProcessMonitor.h"


using namespace lldb_private;

//------------------------------------------------------------------------------
// Static implementations of ProcessMonitor::ReadMemory and
// ProcessMonitor::WriteMemory.  This enables mutual recursion between these
// functions without needed to go thru the thread funnel.

static size_t
DoReadMemory(lldb::pid_t pid, unsigned word_size,
             lldb::addr_t vm_addr, void *buf, size_t size, Error &error)
{
    unsigned char *dst = static_cast<unsigned char*>(buf);
    size_t bytes_read;
    size_t remainder;
    long data;

    for (bytes_read = 0; bytes_read < size; bytes_read += remainder)
    {
        errno = 0;
        data = ptrace(PTRACE_PEEKDATA, pid, vm_addr, NULL);

        if (data == -1L && errno)
        {
            error.SetErrorToErrno();
            return bytes_read;
        }

        remainder = size - bytes_read;
        remainder = remainder > word_size ? word_size : remainder;
        for (unsigned i = 0; i < remainder; ++i)
            dst[i] = ((data >> i*8) & 0xFF);
        vm_addr += word_size;
        dst += word_size;
    }

    return bytes_read;
}

static size_t
DoWriteMemory(lldb::pid_t pid, unsigned word_size,
              lldb::addr_t vm_addr, const void *buf, size_t size, Error &error)
{
    const unsigned char *src = static_cast<const unsigned char*>(buf);
    size_t bytes_written = 0;
    size_t remainder;

    for (bytes_written = 0; bytes_written < size; bytes_written += remainder)
    {
        remainder = size - bytes_written;
        remainder = remainder > word_size ? word_size : remainder;

        if (remainder == word_size)
        {
            unsigned long data = 0;
            for (unsigned i = 0; i < word_size; ++i)
                data |= (unsigned long)src[i] << i*8;

            if (ptrace(PTRACE_POKEDATA, pid, vm_addr, data))
            {
                error.SetErrorToErrno();
                return bytes_written;
            }
        }
        else
        {
            unsigned char buff[8];
            if (DoReadMemory(pid, word_size, vm_addr,
                             buff, word_size, error) != word_size)
                return bytes_written;

            memcpy(buff, src, remainder);

            if (DoWriteMemory(pid, word_size, vm_addr,
                              buff, word_size, error) != word_size)
                return bytes_written;
        }

        vm_addr += word_size;
        src += word_size;
    }
    return bytes_written;
}


//------------------------------------------------------------------------------
/// @class Operation
/// @brief Represents a ProcessMonitor operation.
///
/// Under Linux, it is not possible to ptrace() from any other thread but the
/// one that spawned or attached to the process from the start.  Therefore, when
/// a ProcessMonitor is asked to deliver or change the state of an inferior
/// process the operation must be "funneled" to a specific thread to perform the
/// task.  The Operation class provides an abstract base for all services the
/// ProcessMonitor must perform via the single virtual function Execute, thus
/// encapsulating the code that needs to run in the privileged context.
class Operation
{
public:
    virtual void Execute(ProcessMonitor *monitor) = 0;
};

//------------------------------------------------------------------------------
/// @class ReadOperation
/// @brief Implements ProcessMonitor::ReadMemory.
class ReadOperation : public Operation
{
public:
    ReadOperation(lldb::addr_t addr, void *buff, size_t size,
                  Error &error, size_t &result)
        : m_addr(addr), m_buff(buff), m_size(size),
          m_error(error), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::addr_t m_addr;
    void *m_buff;
    size_t m_size;
    Error &m_error;
    size_t &m_result;
};

void
ReadOperation::Execute(ProcessMonitor *monitor)
{
    const unsigned word_size = monitor->GetProcess().GetAddressByteSize();
    lldb::pid_t pid = monitor->GetPID();

    m_result = DoReadMemory(pid, word_size, m_addr, m_buff, m_size, m_error);
}

//------------------------------------------------------------------------------
/// @class ReadOperation
/// @brief Implements ProcessMonitor::WriteMemory.
class WriteOperation : public Operation
{
public:
    WriteOperation(lldb::addr_t addr, const void *buff, size_t size,
                   Error &error, size_t &result)
        : m_addr(addr), m_buff(buff), m_size(size),
          m_error(error), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::addr_t m_addr;
    const void *m_buff;
    size_t m_size;
    Error &m_error;
    size_t &m_result;
};

void
WriteOperation::Execute(ProcessMonitor *monitor)
{
    const unsigned word_size = monitor->GetProcess().GetAddressByteSize();
    lldb::pid_t pid = monitor->GetPID();

    m_result = DoWriteMemory(pid, word_size, m_addr, m_buff, m_size, m_error);
}

//------------------------------------------------------------------------------
/// @class ReadRegOperation
/// @brief Implements ProcessMonitor::ReadRegisterValue.
class ReadRegOperation : public Operation
{
public:
    ReadRegOperation(unsigned offset, Scalar &value, bool &result)
        : m_offset(offset), m_value(value), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    unsigned m_offset;
    Scalar &m_value;
    bool &m_result;
};

void
ReadRegOperation::Execute(ProcessMonitor *monitor)
{
    lldb::pid_t pid = monitor->GetPID();

    // Set errno to zero so that we can detect a failed peek.
    errno = 0;
    unsigned long data = ptrace(PTRACE_PEEKUSER, pid, m_offset, NULL);

    if (data == -1UL && errno)
        m_result = false;
    else
    {
        m_value = data;
        m_result = true;
    }
}

//------------------------------------------------------------------------------
/// @class WriteRegOperation
/// @brief Implements ProcessMonitor::WriteRegisterValue.
class WriteRegOperation : public Operation
{
public:
    WriteRegOperation(unsigned offset, const Scalar &value, bool &result)
        : m_offset(offset), m_value(value), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    unsigned m_offset;
    const Scalar &m_value;
    bool &m_result;
};

void
WriteRegOperation::Execute(ProcessMonitor *monitor)
{
    lldb::pid_t pid = monitor->GetPID();

    if (ptrace(PTRACE_POKEUSER, pid, m_offset, m_value.ULong()))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class ReadGPROperation
/// @brief Implements ProcessMonitor::ReadGPR.
class ReadGPROperation : public Operation
{
public:
    ReadGPROperation(void *buf, bool &result)
        : m_buf(buf), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    void *m_buf;
    bool &m_result;
};

void
ReadGPROperation::Execute(ProcessMonitor *monitor)
{
    if (ptrace(PTRACE_GETREGS, monitor->GetPID(), NULL, m_buf) < 0)
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class ReadFPROperation
/// @brief Implements ProcessMonitor::ReadFPR.
class ReadFPROperation : public Operation
{
public:
    ReadFPROperation(void *buf, bool &result)
        : m_buf(buf), m_result(result)
        { }

    void Execute(ProcessMonitor *monitor);

private:
    void *m_buf;
    bool &m_result;
};

void
ReadFPROperation::Execute(ProcessMonitor *monitor)
{
    if (ptrace(PTRACE_GETFPREGS, monitor->GetPID(), NULL, m_buf) < 0)
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class ResumeOperation
/// @brief Implements ProcessMonitor::Resume.
class ResumeOperation : public Operation
{
public:
    ResumeOperation(lldb::tid_t tid, bool &result) :
        m_tid(tid), m_result(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::tid_t m_tid;
    bool &m_result;
};

void
ResumeOperation::Execute(ProcessMonitor *monitor)
{
    if (ptrace(PTRACE_CONT, m_tid, NULL, NULL))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class ResumeOperation
/// @brief Implements ProcessMonitor::SingleStep.
class SingleStepOperation : public Operation
{
public:
    SingleStepOperation(lldb::tid_t tid, bool &result)
        : m_tid(tid), m_result(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::tid_t m_tid;
    bool &m_result;
};

void
SingleStepOperation::Execute(ProcessMonitor *monitor)
{
    if (ptrace(PTRACE_SINGLESTEP, m_tid, NULL, NULL))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class SiginfoOperation
/// @brief Implements ProcessMonitor::GetSignalInfo.
class SiginfoOperation : public Operation
{
public:
    SiginfoOperation(lldb::tid_t tid, void *info, bool &result)
        : m_tid(tid), m_info(info), m_result(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::tid_t m_tid;
    void *m_info;
    bool &m_result;
};

void
SiginfoOperation::Execute(ProcessMonitor *monitor)
{
    if (ptrace(PTRACE_GETSIGINFO, m_tid, NULL, m_info))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class EventMessageOperation
/// @brief Implements ProcessMonitor::GetEventMessage.
class EventMessageOperation : public Operation
{
public:
    EventMessageOperation(lldb::tid_t tid, unsigned long *message, bool &result)
        : m_tid(tid), m_message(message), m_result(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    lldb::tid_t m_tid;
    unsigned long *m_message;
    bool &m_result;
};

void
EventMessageOperation::Execute(ProcessMonitor *monitor)
{
    if (ptrace(PTRACE_GETEVENTMSG, m_tid, NULL, m_message))
        m_result = false;
    else
        m_result = true;
}

//------------------------------------------------------------------------------
/// @class KillOperation
/// @brief Implements ProcessMonitor::BringProcessIntoLimbo.
class KillOperation : public Operation
{
public:
    KillOperation(bool &result) : m_result(result) { }

    void Execute(ProcessMonitor *monitor);

private:
    bool &m_result;
};

void
KillOperation::Execute(ProcessMonitor *monitor)
{
    lldb::pid_t pid = monitor->GetPID();

    if (ptrace(PTRACE_KILL, pid, NULL, NULL))
        m_result = false;
    else
        m_result = true;

#if 0
    // First, stop the inferior process.
    if (kill(pid, SIGSTOP))
    {
        m_result = false;
        return;
    }

    // Clear any ptrace options.  When PTRACE_O_TRACEEXIT is set, a plain
    // PTRACE_KILL (or any termination signal) will not truely terminate the
    // inferior process.  Instead, the process is left in a state of "limbo"
    // allowing us to interrogate its state.  However in this case we really do
    // want the process gone.
    if (ptrace(PTRACE_SETOPTIONS, pid, NULL, 0UL))
    {
        m_result = false;
        return;
    }

    // Kill it.
    if (ptrace(PTRACE_KILL, pid, NULL, NULL))
        m_result = false;
    else
        m_result = true;
#endif
}

ProcessMonitor::LaunchArgs::LaunchArgs(ProcessMonitor *monitor,
                                       lldb_private::Module *module,
                                       char const **argv,
                                       char const **envp,
                                       const char *stdin_path,
                                       const char *stdout_path,
                                       const char *stderr_path)
    : m_monitor(monitor),
      m_module(module),
      m_argv(argv),
      m_envp(envp),
      m_stdin_path(stdin_path),
      m_stdout_path(stdout_path),
      m_stderr_path(stderr_path)
{
    sem_init(&m_semaphore, 0, 0);
}

ProcessMonitor::LaunchArgs::~LaunchArgs()
{
    sem_destroy(&m_semaphore);
}

//------------------------------------------------------------------------------
/// The basic design of the ProcessMonitor is built around two threads.
///
/// One thread (@see SignalThread) simply blocks on a call to waitpid() looking
/// for changes in the debugee state.  When a change is detected a
/// ProcessMessage is sent to the associated ProcessLinux instance.  This thread
/// "drives" state changes in the debugger.
///
/// The second thread (@see OperationThread) is responsible for two things 1)
/// launching or attaching to the inferior process, and then 2) servicing
/// operations such as register reads/writes, stepping, etc.  See the comments
/// on the Operation class for more info as to why this is needed.
ProcessMonitor::ProcessMonitor(ProcessLinux *process,
                               Module *module,
                               const char *argv[],
                               const char *envp[],
                               const char *stdin_path,
                               const char *stdout_path,
                               const char *stderr_path,
                               lldb_private::Error &error)
    : m_process(process),
      m_operation_thread(LLDB_INVALID_HOST_THREAD),
      m_pid(LLDB_INVALID_PROCESS_ID),
      m_terminal_fd(-1),
      m_monitor_thread(LLDB_INVALID_HOST_THREAD),
      m_client_fd(-1),
      m_server_fd(-1)
{
    std::auto_ptr<LaunchArgs> args;

    args.reset(new LaunchArgs(this, module, argv, envp,
                              stdin_path, stdout_path, stderr_path));

    // Server/client descriptors.
    if (!EnableIPC())
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Monitor failed to initialize.");
    }

    StartOperationThread(args.get(), error);
    if (!error.Success())
        return;

WAIT_AGAIN:
    // Wait for the operation thread to initialize.
    if (sem_wait(&args->m_semaphore))
    {
        if (errno == EINTR)
            goto WAIT_AGAIN;
        else
        {
            error.SetErrorToErrno();
            return;
        }
    }

    // Check that the launch was a success.
    if (!args->m_error.Success())
    {
        StopOperationThread();
        error = args->m_error;
        return;
    }

    // Finally, start monitoring the child process for change in state.
    m_monitor_thread = Host::StartMonitoringChildProcess(
        ProcessMonitor::MonitorCallback, this, GetPID(), true);
    if (m_monitor_thread == LLDB_INVALID_HOST_THREAD)
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Process launch failed.");
        return;
    }
}

ProcessMonitor::~ProcessMonitor()
{
    StopMonitoringChildProcess();
    StopOperationThread();

    close(m_terminal_fd);
    close(m_client_fd);
    close(m_server_fd);
}

//------------------------------------------------------------------------------
// Thread setup and tear down.
void
ProcessMonitor::StartOperationThread(LaunchArgs *args, Error &error)
{
    static const char *g_thread_name = "lldb.process.linux.operation";

    if (m_operation_thread != LLDB_INVALID_HOST_THREAD)
        return;

    m_operation_thread =
        Host::ThreadCreate(g_thread_name, OperationThread, args, &error);
}

void
ProcessMonitor::StopOperationThread()
{
    lldb::thread_result_t result;

    if (m_operation_thread == LLDB_INVALID_HOST_THREAD)
        return;

    Host::ThreadCancel(m_operation_thread, NULL);
    Host::ThreadJoin(m_operation_thread, &result, NULL);
}

void *
ProcessMonitor::OperationThread(void *arg)
{
    LaunchArgs *args = static_cast<LaunchArgs*>(arg);

    if (!Launch(args))
        return NULL;

    ServeOperation(args->m_monitor);
    return NULL;
}

bool
ProcessMonitor::Launch(LaunchArgs *args)
{
    ProcessMonitor *monitor = args->m_monitor;
    ProcessLinux &process = monitor->GetProcess();
    const char **argv = args->m_argv;
    const char **envp = args->m_envp;
    const char *stdin_path = args->m_stdin_path;
    const char *stdout_path = args->m_stdout_path;
    const char *stderr_path = args->m_stderr_path;

    lldb_utility::PseudoTerminal terminal;
    const size_t err_len = 1024;
    char err_str[err_len];
    lldb::pid_t pid;

    lldb::ThreadSP inferior;

    // Propagate the environment if one is not supplied.
    if (envp == NULL || envp[0] == NULL)
        envp = const_cast<const char **>(environ);

    // Pseudo terminal setup.
    if (!terminal.OpenFirstAvailableMaster(O_RDWR | O_NOCTTY, err_str, err_len))
    {
        args->m_error.SetErrorToGenericError();
        args->m_error.SetErrorString("Could not open controlling TTY.");
        goto FINISH;
    }

    if ((pid = terminal.Fork(err_str, err_len)) < 0)
    {
        args->m_error.SetErrorToGenericError();
        args->m_error.SetErrorString("Process fork failed.");
        goto FINISH;
    }

    // Child process.
    if (pid == 0)
    {
        // Trace this process.
        ptrace(PTRACE_TRACEME, 0, NULL, NULL);

        // Do not inherit setgid powers.
        setgid(getgid());

        // Let us have our own process group.
        setpgid(0, 0);

        // Dup file descriptors if needed.
        //
        // FIXME: If two or more of the paths are the same we needlessly open
        // the same file multiple times.
        if (stdin_path != NULL && stdin_path[0])
            if (!DupDescriptor(stdin_path, STDIN_FILENO, O_RDONLY | O_CREAT))
                exit(1);

        if (stdout_path != NULL && stdout_path[0])
            if (!DupDescriptor(stdout_path, STDOUT_FILENO, O_WRONLY | O_CREAT))
                exit(1);

        if (stderr_path != NULL && stderr_path[0])
            if (!DupDescriptor(stderr_path, STDOUT_FILENO, O_WRONLY | O_CREAT))
                exit(1);

        // Execute.  We should never return.
        execve(argv[0],
               const_cast<char *const *>(argv),
               const_cast<char *const *>(envp));
        exit(-1);
    }

    // Wait for the child process to to trap on its call to execve.
    int status;
    if ((status = waitpid(pid, NULL, 0)) < 0)
    {
        // execve likely failed for some reason.
        args->m_error.SetErrorToErrno();
        goto FINISH;
    }
    assert(status == pid && "Could not sync with inferior process.");

    // Have the child raise an event on exit.  This is used to keep the child in
    // limbo until it is destroyed.
    if (ptrace(PTRACE_SETOPTIONS, pid, NULL, PTRACE_O_TRACEEXIT) < 0)
    {
        args->m_error.SetErrorToErrno();
        goto FINISH;
    }

    // Release the master terminal descriptor and pass it off to the
    // ProcessMonitor instance.  Similarly stash the inferior pid.
    monitor->m_terminal_fd = terminal.ReleaseMasterFileDescriptor();
    monitor->m_pid = pid;

    // Update the process thread list with this new thread and mark it as
    // current.
    inferior.reset(new LinuxThread(process, pid));
    process.GetThreadList().AddThread(inferior);
    process.GetThreadList().SetSelectedThreadByID(pid);

    // Let our process instance know the thread has stopped.
    process.SendMessage(ProcessMessage::Trace(pid));

FINISH:
    // Sync with our parent thread now that the launch operation is complete.
    sem_post(&args->m_semaphore);
    return args->m_error.Success();
}

bool
ProcessMonitor::EnableIPC()
{
    int fd[2];

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fd))
        return false;

    m_client_fd = fd[0];
    m_server_fd = fd[1];
    return true;
}

bool
ProcessMonitor::MonitorCallback(void *callback_baton,
                                lldb::pid_t pid,
                                int signal,
                                int status)
{
    ProcessMessage message;
    ProcessMonitor *monitor = static_cast<ProcessMonitor*>(callback_baton);
    ProcessLinux *process = monitor->m_process;

    switch (signal)
    {
    case 0:
        // No signal.  The child has exited normally.
        message = ProcessMessage::Exit(pid, status);
        break;

    case SIGTRAP:
        // Specially handle SIGTRAP and form the appropriate message.
        message = MonitorSIGTRAP(monitor, pid);
        break;

    default:
        // For all other signals simply notify the process instance.  Note that
        // the process exit status is set when the signal resulted in
        // termination.
        //
        // FIXME: We need a specialized message to inform the process instance
        // about "crashes".
        if (status)
            message = ProcessMessage::Exit(pid, status);
        else
            message = ProcessMessage::Signal(pid, signal);
    }

    process->SendMessage(message);
    bool stop_monitoring = message.GetKind() == ProcessMessage::eExitMessage;
    return stop_monitoring;
}

ProcessMessage
ProcessMonitor::MonitorSIGTRAP(ProcessMonitor *monitor, lldb::pid_t pid)
{
    siginfo_t info;
    ProcessMessage message;
    bool status;

    status = monitor->GetSignalInfo(pid, &info);
    assert(status && "GetSignalInfo failed!");

    assert(info.si_signo == SIGTRAP && "Unexpected child signal!");

    switch (info.si_code)
    {
    default:
        assert(false && "Unexpected SIGTRAP code!");
        break;

    case (SIGTRAP | (PTRACE_EVENT_EXIT << 8)):
    {
        // The inferior process is about to exit.  Maintain the process in a
        // state of "limbo" until we are explicitly commanded to detach,
        // destroy, resume, etc.
        unsigned long data = 0;
        if (!monitor->GetEventMessage(pid, &data))
            data = -1;
        message = ProcessMessage::Exit(pid, (data >> 8));
        break;
    }

    case 0:
    case TRAP_TRACE:
        message = ProcessMessage::Trace(pid);
        break;

    case SI_KERNEL:
    case TRAP_BRKPT:
        message = ProcessMessage::Break(pid);
        break;
    }

    return message;
}

void
ProcessMonitor::ServeOperation(ProcessMonitor *monitor)
{
    int status;
    pollfd fdset;

    fdset.fd = monitor->m_server_fd;
    fdset.events = POLLIN | POLLPRI;
    fdset.revents = 0;

    for (;;)
    {
        if ((status = poll(&fdset, 1, -1)) < 0)
        {
            switch (errno)
            {
            default:
                assert(false && "Unexpected poll() failure!");
                continue;

            case EINTR: continue; // Just poll again.
            case EBADF: return;   // Connection terminated.
            }
        }

        assert(status == 1 && "Too many descriptors!");

        if (fdset.revents & POLLIN)
        {
            Operation *op = NULL;

        READ_AGAIN:
            if ((status = read(fdset.fd, &op, sizeof(op))) < 0)
            {
                // There is only one acceptable failure.
                assert(errno == EINTR);
                goto READ_AGAIN;
            }

            assert(status == sizeof(op));
            op->Execute(monitor);
            write(fdset.fd, &op, sizeof(op));
        }
    }
}

void
ProcessMonitor::DoOperation(Operation *op)
{
    int status;
    Operation *ack = NULL;
    Mutex::Locker lock(m_server_mutex);

    // FIXME: Do proper error checking here.
    write(m_client_fd, &op, sizeof(op));

READ_AGAIN:
    if ((status = read(m_client_fd, &ack, sizeof(ack))) < 0)
    {
        // If interrupted by a signal handler try again.  Otherwise the monitor
        // thread probably died and we have a stale file descriptor -- abort the
        // operation.
        if (errno == EINTR)
            goto READ_AGAIN;
        return;
    }

    assert(status == sizeof(ack));
    assert(ack == op && "Invalid monitor thread response!");
}

size_t
ProcessMonitor::ReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                           Error &error)
{
    size_t result;
    ReadOperation op(vm_addr, buf, size, error, result);
    DoOperation(&op);
    return result;
}

size_t
ProcessMonitor::WriteMemory(lldb::addr_t vm_addr, const void *buf, size_t size,
                            lldb_private::Error &error)
{
    size_t result;
    WriteOperation op(vm_addr, buf, size, error, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::ReadRegisterValue(unsigned offset, Scalar &value)
{
    bool result;
    ReadRegOperation op(offset, value, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::WriteRegisterValue(unsigned offset, const Scalar &value)
{
    bool result;
    WriteRegOperation op(offset, value, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::ReadGPR(void *buf)
{
    bool result;
    ReadGPROperation op(buf, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::ReadFPR(void *buf)
{
    bool result;
    ReadFPROperation op(buf, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::Resume(lldb::tid_t tid)
{
    bool result;
    ResumeOperation op(tid, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::SingleStep(lldb::tid_t tid)
{
    bool result;
    SingleStepOperation op(tid, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::BringProcessIntoLimbo()
{
    bool result;
    KillOperation op(result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::GetSignalInfo(lldb::tid_t tid, void *siginfo)
{
    bool result;
    SiginfoOperation op(tid, siginfo, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::GetEventMessage(lldb::tid_t tid, unsigned long *message)
{
    bool result;
    EventMessageOperation op(tid, message, result);
    DoOperation(&op);
    return result;
}

bool
ProcessMonitor::DupDescriptor(const char *path, int fd, int flags)
{
    int target_fd = open(path, flags);

    if (target_fd == -1)
        return false;

    return (dup2(fd, target_fd) == -1) ? false : true;
}

void
ProcessMonitor::StopMonitoringChildProcess()
{
    lldb::thread_result_t thread_result;

    if (m_monitor_thread != LLDB_INVALID_HOST_THREAD)
    {
        Host::ThreadCancel(m_monitor_thread, NULL);
        Host::ThreadJoin(m_monitor_thread, &thread_result, NULL);
        m_monitor_thread = LLDB_INVALID_HOST_THREAD;
    }
}
