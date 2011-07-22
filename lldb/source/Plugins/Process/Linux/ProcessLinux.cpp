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
#include "Plugins/Process/Utility/InferiorCallPOSIX.h"
#include "ProcessMonitor.h"
#include "LinuxThread.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------------------
// Static functions.

Process*
ProcessLinux::CreateInstance(Target& target, Listener &listener)
{
    return new ProcessLinux(target, listener);
}

void
ProcessLinux::Initialize()
{
    static bool g_initialized = false;

    if (!g_initialized)
    {
        PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                      GetPluginDescriptionStatic(),
                                      CreateInstance);
        g_initialized = true;
    }
}

void
ProcessLinux::Terminate()
{
}

const char *
ProcessLinux::GetPluginNameStatic()
{
    return "plugin.process.linux";
}

const char *
ProcessLinux::GetPluginDescriptionStatic()
{
    return "Process plugin for Linux";
}


//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessLinux::ProcessLinux(Target& target, Listener &listener)
    : Process(target, listener),
      m_monitor(NULL),
      m_module(NULL),
      m_in_limbo(false),
      m_exit_now(false)
{
    // FIXME: Putting this code in the ctor and saving the byte order in a
    // member variable is a hack to avoid const qual issues in GetByteOrder.
    ObjectFile *obj_file = GetTarget().GetExecutableModule()->GetObjectFile();
    m_byte_order = obj_file->GetByteOrder();
}

ProcessLinux::~ProcessLinux()
{
    delete m_monitor;
}

//------------------------------------------------------------------------------
// Process protocol.

bool
ProcessLinux::CanDebug(Target &target, bool plugin_specified_by_name)
{
    // For now we are just making sure the file exists for a given module
    ModuleSP exe_module_sp(target.GetExecutableModule());
    if (exe_module_sp.get())
        return exe_module_sp->GetFileSpec().Exists();
    return false;
}

Error
ProcessLinux::DoAttachToProcessWithID(lldb::pid_t pid)
{
    Error error;
    assert(m_monitor == NULL);

    m_monitor = new ProcessMonitor(this, pid, error);

    if (!error.Success())
        return error;

    SetID(pid);
    return error;
}

Error
ProcessLinux::WillLaunch(Module* module)
{
    Error error;
    return error;
}

Error
ProcessLinux::DoLaunch(Module *module,
                       char const *argv[],
                       char const *envp[],
                       uint32_t launch_flags,
                       const char *stdin_path,
                       const char *stdout_path,
                       const char *stderr_path,
                       const char *working_directory)
{
    Error error;
    assert(m_monitor == NULL);

    SetPrivateState(eStateLaunching);
    m_monitor = new ProcessMonitor(this, module,
                                   argv, envp,
                                   stdin_path, stdout_path, stderr_path,
                                   error);

    m_module = module;

    if (!error.Success())
        return error;

    SetID(m_monitor->GetPID());
    return error;
}

void
ProcessLinux::DidLaunch()
{
}

Error
ProcessLinux::DoResume()
{
    StateType state = GetPrivateState();

    assert(state == eStateStopped || state == eStateCrashed);

    // We are about to resume a thread that will cause the process to exit so
    // set our exit status now.  Do not change our state if the inferior
    // crashed.
    if (state == eStateStopped) 
    {
        if (m_in_limbo)
            SetExitStatus(m_exit_status, NULL);
        else
            SetPrivateState(eStateRunning);
    }

    bool did_resume = false;
    uint32_t thread_count = m_thread_list.GetSize(false);
    for (uint32_t i = 0; i < thread_count; ++i)
    {
        LinuxThread *thread = static_cast<LinuxThread*>(
            m_thread_list.GetThreadAtIndex(i, false).get());
        did_resume = thread->Resume() || did_resume;
    }
    assert(did_resume && "Process resume failed!");

    return Error();
}

addr_t
ProcessLinux::GetImageInfoAddress()
{
    Target *target = &GetTarget();
    ObjectFile *obj_file = target->GetExecutableModule()->GetObjectFile();
    Address addr = obj_file->GetImageInfoAddress();

    if (addr.IsValid()) 
        return addr.GetLoadAddress(target);
    else
        return LLDB_INVALID_ADDRESS;
}

Error
ProcessLinux::DoHalt(bool &caused_stop)
{
    Error error;

    if (IsStopped())
    {
        caused_stop = false;
    }
    else if (kill(GetID(), SIGSTOP))
    {
        caused_stop = false;
        error.SetErrorToErrno();
    }
    else
    {
        caused_stop = true;
    }

    return error;
}

Error
ProcessLinux::DoDetach()
{
    return Error(1, eErrorTypeGeneric);
}

Error
ProcessLinux::DoSignal(int signal)
{
    Error error;

    if (kill(GetID(), signal))
        error.SetErrorToErrno();

    return error;
}

Error
ProcessLinux::DoDestroy()
{
    Error error;

    if (!HasExited())
    {
        // Drive the exit event to completion (do not keep the inferior in
        // limbo).
        m_exit_now = true;

        if (kill(m_monitor->GetPID(), SIGKILL) && error.Success())
        {
            error.SetErrorToErrno();
            return error;
        }

        SetPrivateState(eStateExited);
    }

    return error;
}

void
ProcessLinux::SendMessage(const ProcessMessage &message)
{
    Mutex::Locker lock(m_message_mutex);

    switch (message.GetKind())
    {
    default:
        assert(false && "Unexpected process message!");
        break;

    case ProcessMessage::eInvalidMessage:
        return;

    case ProcessMessage::eLimboMessage:
        m_in_limbo = true;
        m_exit_status = message.GetExitStatus();
        if (m_exit_now)
        {
            SetPrivateState(eStateExited);
            m_monitor->Detach();
        }
        else
            SetPrivateState(eStateStopped);
        break;

    case ProcessMessage::eExitMessage:
        m_exit_status = message.GetExitStatus();
        SetExitStatus(m_exit_status, NULL);
        break;

    case ProcessMessage::eTraceMessage:
    case ProcessMessage::eBreakpointMessage:
        SetPrivateState(eStateStopped);
        break;

    case ProcessMessage::eSignalMessage:
    case ProcessMessage::eSignalDeliveredMessage:
        SetPrivateState(eStateStopped);
        break;

    case ProcessMessage::eCrashMessage:
        SetPrivateState(eStateCrashed);
        break;
    }

    m_message_queue.push(message);
}

void
ProcessLinux::RefreshStateAfterStop()
{
    Mutex::Locker lock(m_message_mutex);
    if (m_message_queue.empty())
        return;

    ProcessMessage &message = m_message_queue.front();

    // Resolve the thread this message corresponds to and pass it along.
    lldb::tid_t tid = message.GetTID();
    LinuxThread *thread = static_cast<LinuxThread*>(
        GetThreadList().FindThreadByID(tid, false).get());

    thread->Notify(message);

    m_message_queue.pop();
}

bool
ProcessLinux::IsAlive()
{
    StateType state = GetPrivateState();
    return state != eStateExited && state != eStateInvalid;
}

size_t
ProcessLinux::DoReadMemory(addr_t vm_addr,
                           void *buf, size_t size, Error &error)
{
    return m_monitor->ReadMemory(vm_addr, buf, size, error);
}

size_t
ProcessLinux::DoWriteMemory(addr_t vm_addr, const void *buf, size_t size,
                            Error &error)
{
    return m_monitor->WriteMemory(vm_addr, buf, size, error);
}

addr_t
ProcessLinux::DoAllocateMemory(size_t size, uint32_t permissions,
                               Error &error)
{
    addr_t allocated_addr = LLDB_INVALID_ADDRESS;

    unsigned prot = 0;
    if (permissions & lldb::ePermissionsReadable)
        prot |= eMmapProtRead;
    if (permissions & lldb::ePermissionsWritable)
        prot |= eMmapProtWrite;
    if (permissions & lldb::ePermissionsExecutable)
        prot |= eMmapProtExec;

    if (InferiorCallMmap(this, allocated_addr, 0, size, prot,
                         eMmapFlagsAnon | eMmapFlagsPrivate, -1, 0)) {
        m_addr_to_mmap_size[allocated_addr] = size;
        error.Clear();
    } else {
        allocated_addr = LLDB_INVALID_ADDRESS;
        error.SetErrorStringWithFormat("unable to allocate %zu bytes of memory with permissions %s", size, GetPermissionsAsCString (permissions));
    }

    return allocated_addr;
}

Error
ProcessLinux::DoDeallocateMemory(lldb::addr_t addr)
{
    Error error;
    MMapMap::iterator pos = m_addr_to_mmap_size.find(addr);
    if (pos != m_addr_to_mmap_size.end() &&
        InferiorCallMunmap(this, addr, pos->second))
        m_addr_to_mmap_size.erase (pos);
    else
        error.SetErrorStringWithFormat("unable to deallocate memory at 0x%llx", addr);

    return error;
}

size_t
ProcessLinux::GetSoftwareBreakpointTrapOpcode(BreakpointSite* bp_site)
{
    static const uint8_t g_i386_opcode[] = { 0xCC };

    ArchSpec arch = GetTarget().GetArchitecture();
    const uint8_t *opcode = NULL;
    size_t opcode_size = 0;

    switch (arch.GetCore())
    {
    default:
        assert(false && "CPU type not supported!");
        break;

    case ArchSpec::eCore_x86_32_i386:
    case ArchSpec::eCore_x86_64_x86_64:
        opcode = g_i386_opcode;
        opcode_size = sizeof(g_i386_opcode);
        break;
    }

    bp_site->SetTrapOpcode(opcode, opcode_size);
    return opcode_size;
}

Error
ProcessLinux::EnableBreakpoint(BreakpointSite *bp_site)
{
    return EnableSoftwareBreakpoint(bp_site);
}

Error
ProcessLinux::DisableBreakpoint(BreakpointSite *bp_site)
{
    return DisableSoftwareBreakpoint(bp_site);
}

uint32_t
ProcessLinux::UpdateThreadListIfNeeded()
{
    // Do not allow recursive updates.
    return m_thread_list.GetSize(false);
}

ByteOrder
ProcessLinux::GetByteOrder() const
{
    // FIXME: We should be able to extract this value directly.  See comment in
    // ProcessLinux().
    return m_byte_order;
}

size_t
ProcessLinux::PutSTDIN(const char *buf, size_t len, Error &error)
{
    ssize_t status;
    if ((status = write(m_monitor->GetTerminalFD(), buf, len)) < 0) 
    {
        error.SetErrorToErrno();
        return 0;
    }
    return status;
}

size_t
ProcessLinux::GetSTDOUT(char *buf, size_t len, Error &error)
{
    ssize_t bytes_read;

    // The terminal file descriptor is always in non-block mode.
    if ((bytes_read = read(m_monitor->GetTerminalFD(), buf, len)) < 0) 
    {
        if (errno != EAGAIN)
            error.SetErrorToErrno();
        return 0;
    }
    return bytes_read;
}

size_t
ProcessLinux::GetSTDERR(char *buf, size_t len, Error &error)
{
    return GetSTDOUT(buf, len, error);
}

UnixSignals &
ProcessLinux::GetUnixSignals()
{
    return m_linux_signals;
}

//------------------------------------------------------------------------------
// ProcessInterface protocol.

const char *
ProcessLinux::GetPluginName()
{
    return "process.linux";
}

const char *
ProcessLinux::GetShortPluginName()
{
    return "process.linux";
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

//------------------------------------------------------------------------------
// Utility functions.

bool
ProcessLinux::HasExited()
{
    switch (GetPrivateState())
    {
    default:
        break;

    case eStateDetached:
    case eStateExited:
        return true;
    }

    return false;
}

bool
ProcessLinux::IsStopped()
{
    switch (GetPrivateState())
    {
    default:
        break;

    case eStateStopped:
    case eStateCrashed:
    case eStateSuspended:
        return true;
    }

    return false;
}
