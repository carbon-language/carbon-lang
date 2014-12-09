//===-- ProcessWindows.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Windows includes
#include "lldb/Host/windows/windows.h"

// C++ Includes
#include <list>
#include <vector>

// Other libraries and framework includes
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/HostNativeProcessBase.h"
#include "lldb/Host/HostNativeThreadBase.h"
#include "lldb/Host/MonitoringProcessLauncher.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/ProcessLauncherWindows.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/FileAction.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"

#include "DebuggerThread.h"
#include "ExceptionRecord.h"
#include "LocalDebugDelegate.h"
#include "ProcessWindows.h"
#include "TargetThreadWindows.h"

#include "llvm/Support/raw_ostream.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private
{

// We store a pointer to this class in the ProcessWindows, so that we don't expose Windows
// OS specific types and implementation details from a public header file.
class ProcessWindowsData
{
  public:
    ProcessWindowsData(const ProcessLaunchInfo &launch_info)
        : m_initial_stop_event(nullptr)
        , m_launch_info(launch_info)
        , m_initial_stop_received(false)
    {
        m_initial_stop_event = ::CreateEvent(nullptr, TRUE, FALSE, nullptr);
    }

    ~ProcessWindowsData() { ::CloseHandle(m_initial_stop_event); }

    ProcessLaunchInfo m_launch_info;
    lldb_private::Error m_launch_error;
    lldb_private::DebuggerThreadSP m_debugger;
    StopInfoSP m_pending_stop_info;
    HANDLE m_initial_stop_event;
    bool m_initial_stop_received;
    std::map<lldb::tid_t, HostThread> m_new_threads;
    std::map<lldb::tid_t, HostThread> m_exited_threads;
};
}
//------------------------------------------------------------------------------
// Static functions.

ProcessSP
ProcessWindows::CreateInstance(Target &target, Listener &listener, const FileSpec *)
{
    return ProcessSP(new ProcessWindows(target, listener));
}

void
ProcessWindows::Initialize()
{
    static bool g_initialized = false;

    if (!g_initialized)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                      GetPluginDescriptionStatic(),
                                      CreateInstance);
    }
}

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessWindows::ProcessWindows(Target &target, Listener &listener)
    : lldb_private::Process(target, listener)
{
}

ProcessWindows::~ProcessWindows()
{
}

void
ProcessWindows::Terminate()
{
}

lldb_private::ConstString
ProcessWindows::GetPluginNameStatic()
{
    static ConstString g_name("windows");
    return g_name;
}

const char *
ProcessWindows::GetPluginDescriptionStatic()
{
    return "Process plugin for Windows";
}

size_t
ProcessWindows::GetSTDOUT(char *buf, size_t buf_size, Error &error)
{
    error.SetErrorString("GetSTDOUT unsupported on Windows");
    return 0;
}

size_t
ProcessWindows::GetSTDERR(char *buf, size_t buf_size, Error &error)
{
    error.SetErrorString("GetSTDERR unsupported on Windows");
    return 0;
}

size_t
ProcessWindows::PutSTDIN(const char *buf, size_t buf_size, Error &error)
{
    error.SetErrorString("PutSTDIN unsupported on Windows");
    return 0;
}

Error
ProcessWindows::EnableBreakpointSite(BreakpointSite *bp_site)
{
    return EnableSoftwareBreakpoint(bp_site);
}

Error
ProcessWindows::DisableBreakpointSite(BreakpointSite *bp_site)
{
    return DisableSoftwareBreakpoint(bp_site);
}

bool
ProcessWindows::UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    // Add all the threads that were previously running and for which we did not detect a thread
    // exited event.
    int new_size = 0;
    for (ThreadSP old_thread : old_thread_list.Threads())
    {
        lldb::tid_t old_thread_id = old_thread->GetID();
        auto exited_thread_iter = m_session_data->m_exited_threads.find(old_thread_id);
        if (exited_thread_iter == m_session_data->m_exited_threads.end())
        {
            new_thread_list.AddThread(old_thread);
            ++new_size;
        }
    }

    // Also add all the threads that are new since the last time we broke into the debugger.
    for (auto iter = m_session_data->m_new_threads.begin(); iter != m_session_data->m_new_threads.end(); ++iter)
    {
        ThreadSP thread(new TargetThreadWindows(*this, iter->second));
        thread->SetID(iter->first);
        new_thread_list.AddThread(thread);
        ++new_size;
    }

    m_session_data->m_new_threads.clear();
    m_session_data->m_exited_threads.clear();

    return new_size > 0;
}

Error
ProcessWindows::DoLaunch(Module *exe_module,
                         ProcessLaunchInfo &launch_info)
{
    Error result;
    if (!launch_info.GetFlags().Test(eLaunchFlagDebug))
    {
        result.SetErrorString("ProcessWindows can only be used to launch processes for debugging.");
        return result;
    }

    m_session_data.reset(new ProcessWindowsData(launch_info));

    SetPrivateState(eStateLaunching);
    DebugDelegateSP delegate(new LocalDebugDelegate(shared_from_this()));
    m_session_data->m_debugger.reset(new DebuggerThread(delegate));
    DebuggerThreadSP debugger = m_session_data->m_debugger;

    // Kick off the DebugLaunch asynchronously and wait for it to complete.
    result = debugger->DebugLaunch(launch_info);

    HostProcess process;
    if (result.Success())
    {
        // Block this function until we receive the initial stop from the process.
        if (::WaitForSingleObject(m_session_data->m_initial_stop_event, INFINITE) == WAIT_OBJECT_0)
            process = debugger->GetProcess();
        else
            result.SetError(::GetLastError(), eErrorTypeWin32);
    }

    if (!result.Success())
        return result;

    // We've hit the initial stop.  The private state should already be set to stopped as a result
    // of encountering the breakpoint exception in ProcessWindows::OnDebugException.
    launch_info.SetProcessID(process.GetProcessId());
    SetID(process.GetProcessId());

    return result;
}

Error
ProcessWindows::DoResume()
{
    Error error;
    if (GetPrivateState() == eStateStopped || GetPrivateState() == eStateCrashed)
    {
        if (m_session_data->m_debugger->GetActiveException())
        {
            // Resume the process and continue processing debug events.  Mask the exception so that
            // from the process's view, there is no indication that anything happened.
            m_session_data->m_debugger->ContinueAsyncException(ExceptionResult::MaskException);
        }

        SetPrivateState(eStateRunning);
    }
    return error;
}


//------------------------------------------------------------------------------
// ProcessInterface protocol.

lldb_private::ConstString
ProcessWindows::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessWindows::GetPluginVersion()
{
    return 1;
}

Error
ProcessWindows::DoDetach(bool keep_stopped)
{
    Error error;
    return error;
}

Error
ProcessWindows::DoDestroy()
{
    Error error;
    if (GetPrivateState() != eStateExited && GetPrivateState() != eStateDetached && m_session_data)
    {
        DebuggerThread &debugger = *m_session_data->m_debugger;
        error = debugger.StopDebugging(true);
    }
    m_session_data.reset();
    return error;
}

void
ProcessWindows::RefreshStateAfterStop()
{
    m_thread_list.RefreshStateAfterStop();

    ExceptionRecord *active_exception = m_session_data->m_debugger->GetActiveException();
    if (!active_exception)
        return;

    StopInfoSP stop_info;
    ThreadSP stop_thread = m_thread_list.GetSelectedThread();
    RegisterContextSP register_context = stop_thread->GetRegisterContext();

    uint64_t pc = register_context->GetPC();
    if (active_exception->GetExceptionCode() == EXCEPTION_BREAKPOINT)
    {
        // TODO(zturner): The current EIP is AFTER the BP opcode, which is one byte.  So
        // to find the breakpoint, move the PC back.  A better way to do this is probably
        // to ask the Platform how big a breakpoint opcode is.
        --pc;
        BreakpointSiteSP site(GetBreakpointSiteList().FindByAddress(pc));
        lldb::break_id_t break_id = LLDB_INVALID_BREAK_ID;
        bool should_stop = true;
        if (site)
        {
            should_stop = site->ValidForThisThread(stop_thread.get());
            break_id = site->GetID();
        }

        stop_info = StopInfo::CreateStopReasonWithBreakpointSiteID(*stop_thread, break_id, should_stop);
        stop_thread->SetStopInfo(stop_info);
    }
    else
    {
        std::string desc;
        llvm::raw_string_ostream desc_stream(desc);
        desc_stream << "Exception " << active_exception->GetExceptionCode() << " encountered at address " << pc;
        stop_info = StopInfo::CreateStopReasonWithException(*stop_thread, desc_stream.str().c_str());
        stop_thread->SetStopInfo(stop_info);
    }
}

bool
ProcessWindows::IsAlive()
{
    StateType state = GetPrivateState();
    switch (state)
    {
        case eStateCrashed:
        case eStateDetached:
        case eStateUnloaded:
        case eStateExited:
        case eStateInvalid:
            return false;
        default:
            return true;
    }
}

Error
ProcessWindows::DoHalt(bool &caused_stop)
{
    Error error;
    StateType state = GetPrivateState();
    if (state == eStateStopped)
        caused_stop = false;
    else
    {
        caused_stop = ::DebugBreakProcess(m_session_data->m_debugger->GetProcess().GetNativeProcess().GetSystemHandle());
        if (!caused_stop)
            error.SetError(GetLastError(), eErrorTypeWin32);
    }
    return error;
}

void ProcessWindows::DidLaunch()
{
    StateType state = GetPrivateState();
    // The initial stop won't broadcast the state change event, so account for that here.
    if (m_session_data && GetPrivateState() == eStateStopped &&
            m_session_data->m_launch_info.GetFlags().Test(eLaunchFlagStopAtEntry))
        RefreshStateAfterStop();
}

size_t
ProcessWindows::DoReadMemory(lldb::addr_t vm_addr,
                             void *buf,
                             size_t size,
                             Error &error)
{
    if (!m_session_data)
        return 0;

    HostProcess process = m_session_data->m_debugger->GetProcess();
    void *addr = reinterpret_cast<void *>(vm_addr);
    SIZE_T bytes_read = 0;
    if (!ReadProcessMemory(process.GetNativeProcess().GetSystemHandle(), addr, buf, size, &bytes_read))
        error.SetError(GetLastError(), eErrorTypeWin32);
    return bytes_read;
}

size_t
ProcessWindows::DoWriteMemory(lldb::addr_t vm_addr, const void *buf, size_t size, Error &error)
{
    if (!m_session_data)
        return 0;

    HostProcess process = m_session_data->m_debugger->GetProcess();
    void *addr = reinterpret_cast<void *>(vm_addr);
    SIZE_T bytes_written = 0;
    lldb::process_t handle = process.GetNativeProcess().GetSystemHandle();
    if (WriteProcessMemory(handle, addr, buf, size, &bytes_written))
        FlushInstructionCache(handle, addr, bytes_written);
    else
        error.SetError(GetLastError(), eErrorTypeWin32);
    return bytes_written;
}

lldb::addr_t
ProcessWindows::GetImageInfoAddress()
{
    Target &target = GetTarget();
    ObjectFile *obj_file = target.GetExecutableModule()->GetObjectFile();
    Address addr = obj_file->GetImageInfoAddress(&target);
    if (addr.IsValid())
        return addr.GetLoadAddress(&target);
    else
        return LLDB_INVALID_ADDRESS;
}

bool
ProcessWindows::CanDebug(Target &target, bool plugin_specified_by_name)
{
    if (plugin_specified_by_name)
        return true;

    // For now we are just making sure the file exists for a given module
    ModuleSP exe_module_sp(target.GetExecutableModule());
    if (exe_module_sp.get())
        return exe_module_sp->GetFileSpec().Exists();
    return false;
}

void
ProcessWindows::OnExitProcess(uint32_t exit_code)
{
    ModuleSP executable_module = GetTarget().GetExecutableModule();
    ModuleList unloaded_modules;
    unloaded_modules.Append(executable_module);
    GetTarget().ModulesDidUnload(unloaded_modules, true);

    SetProcessExitStatus(nullptr, GetID(), true, 0, exit_code);
    SetPrivateState(eStateExited);
}

void
ProcessWindows::OnDebuggerConnected(lldb::addr_t image_base)
{
    // Either we successfully attached to an existing process, or we successfully launched a new
    // process under the debugger.
    ModuleSP module = GetTarget().GetExecutableModule();
    bool load_addr_changed;
    module->SetLoadAddress(GetTarget(), image_base, false, load_addr_changed);

    // Notify the target that the executable module has loaded.  This will cause any pending
    // breakpoints to be resolved to explicit brekapoint sites.
    ModuleList loaded_modules;
    loaded_modules.Append(module);
    GetTarget().ModulesDidLoad(loaded_modules);

    DebuggerThreadSP debugger = m_session_data->m_debugger;
    const HostThreadWindows &wmain_thread = debugger->GetMainThread().GetNativeThread();
    m_session_data->m_new_threads[wmain_thread.GetThreadId()] = debugger->GetMainThread();
}

ExceptionResult
ProcessWindows::OnDebugException(bool first_chance, const ExceptionRecord &record)
{
    ExceptionResult result = ExceptionResult::SendToApplication;
    switch (record.GetExceptionCode())
    {
        case EXCEPTION_BREAKPOINT:
            // Handle breakpoints at the first chance.
            result = ExceptionResult::BreakInDebugger;

            if (!m_session_data->m_initial_stop_received)
            {
                m_session_data->m_initial_stop_received = true;
                ::SetEvent(m_session_data->m_initial_stop_event);
            }

            break;
        default:
            // For non-breakpoints, give the application a chance to handle the exception first.
            if (first_chance)
                result = ExceptionResult::SendToApplication;
            else
                result = ExceptionResult::BreakInDebugger;
    }

    if (!first_chance)
    {
        // Any second chance exception is an application crash by definition.
        SetPrivateState(eStateCrashed);
    }
    else if (result == ExceptionResult::BreakInDebugger)
    {
        // For first chance exceptions that we can handle, the process is stopped so the user
        // can interact with the debugger.
        SetPrivateState(eStateStopped);
    }
    else
    {
        // For first chance exceptions that we either eat or send back to the application, don't
        // modify the state of the application.
    }

    return result;
}

void
ProcessWindows::OnCreateThread(const HostThread &new_thread)
{
    const HostThreadWindows &wnew_thread = new_thread.GetNativeThread();
    m_session_data->m_new_threads[wnew_thread.GetThreadId()] = new_thread;
}

void
ProcessWindows::OnExitThread(const HostThread &exited_thread)
{
    // A thread may have started and exited before the debugger stopped allowing a refresh.
    // Just remove it from the new threads list in that case.
    const HostThreadWindows &wexited_thread = exited_thread.GetNativeThread();
    auto iter = m_session_data->m_new_threads.find(wexited_thread.GetThreadId());
    if (iter != m_session_data->m_new_threads.end())
        m_session_data->m_new_threads.erase(iter);
    else
        m_session_data->m_exited_threads[wexited_thread.GetThreadId()] = exited_thread;
}

void
ProcessWindows::OnLoadDll(const ModuleSpec &module_spec, lldb::addr_t module_addr)
{
    // Confusingly, there is no Target::AddSharedModule.  Instead, calling GetSharedModule() with
    // a new module will add it to the module list and return a corresponding ModuleSP.
    Error error;
    ModuleSP module = GetTarget().GetSharedModule(module_spec, &error);
    bool load_addr_changed = false;
    module->SetLoadAddress(GetTarget(), module_addr, false, load_addr_changed);

    ModuleList loaded_modules;
    loaded_modules.Append(module);
    GetTarget().ModulesDidLoad(loaded_modules);
}

void
ProcessWindows::OnUnloadDll(lldb::addr_t module_addr)
{
    Address resolved_addr;
    if (GetTarget().ResolveLoadAddress(module_addr, resolved_addr))
    {
        ModuleSP module = resolved_addr.GetModule();
        if (module)
        {
            ModuleList unloaded_modules;
            unloaded_modules.Append(module);
            GetTarget().ModulesDidUnload(unloaded_modules, false);
        }
    }
}

void
ProcessWindows::OnDebugString(const std::string &string)
{
}

void
ProcessWindows::OnDebuggerError(const Error &error, uint32_t type)
{
    if (!m_session_data->m_initial_stop_received)
    {
        // If we haven't actually launched the process yet, this was an error launching the
        // process.  Set the internal error and signal the initial stop event so that the DoLaunch
        // method wakes up and returns a failure.
        m_session_data->m_launch_error = error;
        ::SetEvent(m_session_data->m_initial_stop_event);
        return;
    }

    // This happened while debugging.  Do we shutdown the debugging session, try to continue,
    // or do something else?
}
