//===-- ProcessMacOSXRemote.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//----------------------------------------------------------------------
//
//  ProcessMacOSXRemote.cpp
//  liblldb
//
//  Created by Greg Clayton on 4/21/09.
//
//
//----------------------------------------------------------------------

// C Includes
#include <errno.h>

// C++ Includes
//#include <algorithm>
//#include <map>

// Other libraries and framework includes

// Project includes
#include "ProcessMacOSXRemote.h"
#include "ProcessMacOSXLog.h"
#include "ThreadMacOSX.h"

Process*
ProcessMacOSXRemote::CreateInstance (Target &target)
{
    return new ProcessMacOSXRemote (target);
}

bool
ProcessMacOSXRemote::CanDebug(Target &target)
{
    // For now we are just making sure the file exists for a given module
    ModuleSP exe_module_sp(target.GetExecutableModule());
    if (exe_module_sp.get())
        return exe_module_sp->GetFileSpec().Exists();
    return false;
}

//----------------------------------------------------------------------
// ProcessMacOSXRemote constructor
//----------------------------------------------------------------------
ProcessMacOSXRemote::ProcessMacOSXRemote(Target& target) :
    Process (target),
    m_flags (0),
    m_arch_spec (),
    m_dynamic_loader_ap (),
    m_byte_order(eByteOrderInvalid)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ProcessMacOSXRemote::~DCProcessMacOSXRemote()
{
    Clear();
}

//----------------------------------------------------------------------
// Process Control
//----------------------------------------------------------------------
lldb::pid_t
ProcessMacOSXRemote::DoLaunch
(
    Module* module,
    char const *argv[],
    char const *envp[],
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path
)
{
//  ::LogSetBitMask (PD_LOG_DEFAULT);
//  ::LogSetOptions (LLDB_LOG_OPTION_THREADSAFE | LLDB_LOG_OPTION_PREPEND_TIMESTAMP | LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD);
//  ::LogSetLogFile ("/dev/stdout");

    ObjectFile * object_file = module->GetObjectFile();
    if (object_file)
    {
        char exec_file_path[PATH_MAX];
        FileSpec* file_spec_ptr = object_file->GetFileSpec();
        if (file_spec_ptr)
            file_spec_ptr->GetPath(exec_file_path, sizeof(exec_file_path));

        ArchSpec arch_spec(module->GetArchitecture());

        switch (arch_spec.GetCPUType())
        {

        }
        // Set our user ID to our process ID.
        SetID(LaunchForDebug(exec_file_path, argv, envp, arch_spec, stdin_path, stdout_path, stderr_path, eLaunchDefault, GetError()));
    }
    else
    {
        // Set our user ID to an invalid process ID.
        SetID(LLDB_INVALID_PROCESS_ID);
        GetError().SetErrorToGenericError ();
        GetError().SetErrorStringWithFormat ("Failed to get object file from '%s' for arch %s.\n", module->GetFileSpec().GetFilename().AsCString(), module->GetArchitecture().AsCString());
    }

    // Return the process ID we have
    return GetID();
}

lldb::pid_t
ProcessMacOSXRemote::DoAttach (lldb::pid_t attach_pid)
{
    // Set our user ID to the attached process ID (which can be invalid if
    // the attach fails
    lldb::pid_t pid = AttachForDebug(attach_pid);
    SetID(pid);

//  if (pid != LLDB_INVALID_PROCESS_ID)
//  {
//      // Wait for a process stopped event, but don't consume it
//      if (WaitForEvents(LLDB_EVENT_STOPPED, NULL, 30))
//      {
//      }
//  }
//
    // Return the process ID we have
    return pid;
}


void
ProcessMacOSXRemote::DidLaunch ()
{
    if (GetID() == LLDB_INVALID_PROCESS_ID)
    {
        m_dynamic_loader_ap.reset();
    }
    else
    {
        Module * exe_module = GetTarget().GetExecutableModule ().get();
        assert(exe_module);
        ObjectFile *exe_objfile = exe_module->GetObjectFile();
        assert(exe_objfile);
        m_byte_order = exe_objfile->GetByteOrder();
        assert(m_byte_order != eByteOrderInvalid);
        // Install a signal handler so we can catch when our child process
        // dies and set the exit status correctly.
        m_wait_thread = Host::ThreadCreate (ProcessMacOSXRemote::WaitForChildProcessToExit, &m_uid, &m_error);
        if (m_wait_thread != LLDB_INVALID_HOST_THREAD)
        {
            // Don't need to get the return value of this thread, so just let
            // it clean up after itself when it dies.
            Host::ThreadDetach (m_wait_thread, NULL);
        }
        m_dynamic_loader_ap.reset(DynamicLoader::FindPlugin(this, "macosx-dyld"));
    }

}

void
ProcessMacOSXRemote::DidAttach ()
{
    DidLaunch ();
    m_need_to_run_did_attach = true;
}

bool
ProcessMacOSXRemote::DoResume ()
{
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSXRemote::Resume()");
    State state = GetState();

    if (CanResume(state))
    {
        PrivateResume(LLDB_INVALID_THREAD_ID);
    }
    else if (state == eStateRunning)
    {
        ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "Resume() - task 0x%x is running, ignoring...", m_task.TaskPort());
        GetError().Clear();

    }
    else
    {
        ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "Resume() - task 0x%x can't continue, ignoring...", m_task.TaskPort());
        GetError().SetError(UINT_MAX, Error::Generic);
    }

    return GetError().Success();
}

size_t
ProcessMacOSXRemote::GetSoftwareBreakpointTrapOpcode (BreakpointSite *bp_site)
{
    ModuleSP exe_module_sp(GetTarget().GetExecutableModule());
    if (exe_module_sp.get())
    {
        const ArchSpec &exe_arch = exe_module_sp->GetArchitecture();
        const uint8_t *trap_opcode = NULL;
        uint32_t trap_opcode_size = 0;

        static const uint8_t g_arm_breakpoint_opcode[] = { 0xFE, 0xDE, 0xFF, 0xE7 };
        //static const uint8_t g_thumb_breakpooint_opcode[] = { 0xFE, 0xDE };
        static const uint8_t g_ppc_breakpoint_opcode[] = { 0x7F, 0xC0, 0x00, 0x08 };
        static const uint8_t g_i386_breakpoint_opcode[] = { 0xCC };

        switch (exe_arch.GetCPUType())
        {
        case CPU_TYPE_ARM:
            // TODO: fill this in for ARM. We need to dig up the symbol for
            // the address in the breakpoint location and figure out if it is
            // an ARM or Thumb breakpoint.
            trap_opcode = g_arm_breakpoint_opcode;
            trap_opcode_size = sizeof(g_arm_breakpoint_opcode);
            break;

        case CPU_TYPE_POWERPC:
        case CPU_TYPE_POWERPC64:
            trap_opcode = g_ppc_breakpoint_opcode;
            trap_opcode_size = sizeof(g_ppc_breakpoint_opcode);
            break;

        case CPU_TYPE_I386:
        case CPU_TYPE_X86_64:
            trap_opcode = g_i386_breakpoint_opcode;
            trap_opcode_size = sizeof(g_i386_breakpoint_opcode);
            break;

        default:
            assert(!"Unhandled architecture in ProcessMacOSXRemote::GetSoftwareBreakpointTrapOpcode()");
            return 0;
        }

        if (trap_opcode && trap_opcode_size)
        {
            if (bp_loc->SetTrapOpcode(trap_opcode, trap_opcode_size))
                return trap_opcode_size;
        }
    }
    // No executable yet, so we can't tell what the breakpoint opcode will be.
    return 0;
}
uint32_t
ProcessMacOSXRemote::UpdateThreadListIfNeeded ()
{
    // locker will keep a mutex locked until it goes out of scope
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_THREAD);
    if (log && log->GetMask().IsSet(PD_LOG_VERBOSE))
        log->Printf ("ProcessMacOSXRemote::%s (pid = %4.4x)", __FUNCTION__, GetID());

    const uint32_t stop_id = GetStopID();
    if (m_thread_list.GetSize() == 0 || stop_id != m_thread_list.GetID())
    {
        m_thread_list.SetID (stop_id);
        thread_array_t thread_list = NULL;
        mach_msg_type_number_t thread_list_count = 0;
        task_t task = Task().TaskPort();
        Error err(::task_threads (task, &thread_list, &thread_list_count), Error::MachKernel);

        if (log || err.Fail())
            err.Log(log, "::task_threads ( task = 0x%4.4x, thread_list => %p, thread_list_count => %u )", task, thread_list, thread_list_count);

        if (err.GetError() == KERN_SUCCESS && thread_list_count > 0)
        {
            ThreadList curr_thread_list;

            size_t idx;
            // Iterator through the current thread list and see which threads
            // we already have in our list (keep them), which ones we don't
            // (add them), and which ones are not around anymore (remove them).
            for (idx = 0; idx < thread_list_count; ++idx)
            {
                const lldb::tid_t tid = thread_list[idx];
                ThreadSP thread_sp(m_thread_list.FindThreadByID (tid));
                if (thread_sp.get() == NULL)
                    thread_sp.reset (new ThreadMacOSX (this, tid));
                curr_thread_list.AddThread(thread_sp);
            }

            m_thread_list = curr_thread_list;

            // Free the vm memory given to us by ::task_threads()
            vm_size_t thread_list_size = (vm_size_t) (thread_list_count * sizeof (lldb::tid_t));
            ::vm_deallocate (::mach_task_self(),
                             (vm_address_t)thread_list,
                             thread_list_size);
        }
    }
    return m_thread_list.GetSize();
}

bool
ProcessMacOSXRemote::ShouldStop ()
{
    // If we are attaching, let our dynamic loader plug-in know so it can get
    // an initial list of shared libraries.
    if (m_need_to_run_did_attach && m_dynamic_loader_ap.get())
    {
        m_need_to_run_did_attach = false;
        m_dynamic_loader_ap->DidAttach();
    }

    // We must be attaching if we don't already have a valid architecture
    if (!m_arch_spec.IsValid())
    {
        Module *exe_module = GetTarget().GetExecutableModule().get();
        if (exe_module)
            m_arch_spec = exe_module->GetArchitecture();
    }
    // Let all threads recover from stopping and do any clean up based
    // on the previous thread state (if any).
    UpdateThreadListIfNeeded ();

    if (m_thread_list.ShouldStop())
    {
        // Let each thread know of any exceptions
        Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_EXCEPTIONS);
        task_t task = m_task.TaskPort();
        size_t i;
        for (i=0; i<m_exception_messages.size(); ++i)
        {
            // Let the thread list figure use the ProcessMacOSXRemote to forward all exceptions
            // on down to each thread.
            if (m_exception_messages[i].state.task_port == task)
            {
                ThreadSP thread_sp(m_thread_list.FindThreadByID(m_exception_messages[i].state.thread_port));
                if (thread_sp.get())
                {
                    ThreadMacOSX *macosx_thread = (ThreadMacOSX *)thread_sp.get();
                    macosx_thread->NotifyException (m_exception_messages[i].state);
                }
            }
            if (log)
                m_exception_messages[i].Log(log);
        }
        return true;
    }
    return false;
}

bool
ProcessMacOSXRemote::DoHalt ()
{
    return Kill (SIGINT);
}

bool
ProcessMacOSXRemote::WillDetach ()
{
    State state = GetState();

    if (IsRunning(state))
    {
        m_error.SetErrorToGenericError();
        m_error.SetErrorString("Process must be stopped in order to detach.");
        return false;
    }
    return true;
}

bool
ProcessMacOSXRemote::DoDetach ()
{
    m_use_public_queue = false;
    bool success = Detach();
    m_use_public_queue = true;
    if (success)
        SetState (eStateDetached);
    return success;
}

bool
ProcessMacOSXRemote::DoKill (int signal)
{
    return Kill (signal);
}


//------------------------------------------------------------------
// Thread Queries
//------------------------------------------------------------------

Thread *
ProcessMacOSXRemote::GetCurrentThread ()
{
    return m_thread_list.GetCurrentThread().get();
}

ByteOrder
ProcessMacOSXRemote::GetByteOrder () const
{
    return m_byte_order;
}



//------------------------------------------------------------------
// Process Queries
//------------------------------------------------------------------

bool
ProcessMacOSXRemote::IsAlive ()
{
    return MachTask::IsValid (Task().TaskPort());
}

bool
ProcessMacOSXRemote::IsRunning ()
{
    return LLDB_STATE_IS_RUNNING(GetState());
}

lldb::addr_t
ProcessMacOSXRemote::GetImageInfoAddress()
{
    return Task().GetDYLDAllImageInfosAddress();
}

DynamicLoader *
ProcessMacOSXRemote::GetDynamicLoader()
{
    return m_dynamic_loader_ap.get();
}

//------------------------------------------------------------------
// Process Memory
//------------------------------------------------------------------

size_t
ProcessMacOSXRemote::DoReadMemory (lldb::addr_t addr, void *buf, size_t size)
{
    return Task().ReadMemory(addr, buf, size);
}

size_t
ProcessMacOSXRemote::DoWriteMemory (lldb::addr_t addr, const void *buf, size_t size)
{
    return Task().WriteMemory(addr, buf, size);
}

//------------------------------------------------------------------
// Process STDIO
//------------------------------------------------------------------

size_t
ProcessMacOSXRemote::GetSTDOUT (char *buf, size_t buf_size)
{
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSXRemote::%s (&%p[%u]) ...", __FUNCTION__, buf, buf_size);
    Mutex::Locker locker(m_stdio_mutex);
    size_t bytes_available = m_stdout_data.size();
    if (bytes_available > 0)
    {
        if (bytes_available > buf_size)
        {
            memcpy(buf, m_stdout_data.data(), buf_size);
            m_stdout_data.erase(0, buf_size);
            bytes_available = buf_size;
        }
        else
        {
            memcpy(buf, m_stdout_data.data(), bytes_available);
            m_stdout_data.clear();
        }
    }
    return bytes_available;
}

size_t
ProcessMacOSXRemote::GetSTDERR (char *buf, size_t buf_size)
{
    return 0;
}

bool
ProcessMacOSXRemote::EnableBreakpoint (BreakpointLocation *bp)
{
    assert (bp != NULL);

    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_BREAKPOINTS);
    lldb::user_id_t breakID = bp->GetID();
    lldb::addr_t addr = bp->GetAddress();
    if (bp->IsEnabled())
    {
        if (log)
            log->Printf("ProcessMacOSXRemote::EnableBreakpoint ( breakID = %d ) breakpoint already enabled.", breakID);
        return true;
    }
    else
    {
        if (bp->HardwarePreferred())
        {
            ThreadMacOSX *thread = (ThreadMacOSX *)m_thread_list.FindThreadByID(bp->GetThreadID()).get();
            if (thread)
            {
                bp->SetHardwareIndex (thread->EnableHardwareBreakpoint(bp));
                if (bp->IsHardware())
                {
                    bp->SetEnabled(true);
                    return true;
                }
            }
        }

        const size_t break_op_size = GetSoftwareBreakpointTrapOpcode (bp);
        assert (break_op_size > 0);
        const uint8_t * const break_op = bp->GetTrapOpcodeBytes();

        if (break_op_size > 0)
        {
            // Save the original opcode by reading it
            if (m_task.ReadMemory(addr, bp->GetSavedOpcodeBytes(), break_op_size) == break_op_size)
            {
                // Write a software breakpoint in place of the original opcode
                if (m_task.WriteMemory(addr, break_op, break_op_size) == break_op_size)
                {
                    uint8_t verify_break_op[4];
                    if (m_task.ReadMemory(addr, verify_break_op, break_op_size) == break_op_size)
                    {
                        if (memcmp(break_op, verify_break_op, break_op_size) == 0)
                        {
                            bp->SetEnabled(true);
                            if (log)
                                log->Printf("ProcessMacOSXRemote::EnableBreakpoint ( breakID = %d ) SUCCESS.", breakID, (uint64_t)addr);
                            return true;
                        }
                        else
                        {
                            GetError().SetErrorString("Failed to verify the breakpoint trap in memory.");
                        }
                    }
                    else
                    {
                        GetError().SetErrorString("Unable to read memory to verify breakpoint trap.");
                    }
                }
                else
                {
                    GetError().SetErrorString("Unable to write breakpoint trap to memory.");
                }
            }
            else
            {
                GetError().SetErrorString("Unable to read memory at breakpoint address.");
            }
        }
    }

    if (log)
    {
        const char *err_string = GetError().AsCString();
        log->Printf ("ProcessMacOSXRemote::EnableBreakpoint ( breakID = %d ) error: %s",
                     breakID, err_string ? err_string : "NULL");
    }
    GetError().SetErrorToGenericError();
    return false;
}

bool
ProcessMacOSXRemote::DisableBreakpoint (BreakpointLocation *bp)
{
    assert (bp != NULL);
    lldb::addr_t addr = bp->GetAddress();
    lldb::user_id_t breakID = bp->GetID();
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_BREAKPOINTS);
    if (log)
        log->Printf ("ProcessMacOSXRemote::DisableBreakpoint (breakID = %d) addr = 0x%8.8llx", breakID, (uint64_t)addr);

    if (bp->IsHardware())
    {
        ThreadMacOSX *thread = (ThreadMacOSX *)m_thread_list.FindThreadByID(bp->GetThreadID()).get();
        if (thread)
        {
            if (thread->DisableHardwareBreakpoint(bp))
            {
                bp->SetEnabled(false);
                if (log)
                    log->Printf ("ProcessMacOSXRemote::DisableBreakpoint (breakID = %d) (hardware) => success", breakID);
                return true;
            }
        }
        return false;
    }

    const size_t break_op_size = bp->GetByteSize();
    assert (break_op_size > 0);
    const uint8_t * const break_op = bp->GetTrapOpcodeBytes();
    if (break_op_size > 0)
    {
        // Clear a software breakpoint instruction
        uint8_t curr_break_op[break_op_size];
        bool break_op_found = false;

        // Read the breakpoint opcode
        if (m_task.ReadMemory(addr, curr_break_op, break_op_size) == break_op_size)
        {
            bool verify = false;
            if (bp->IsEnabled())
            {
                // Make sure we have the a breakpoint opcode exists at this address
                if (memcmp(curr_break_op, break_op, break_op_size) == 0)
                {
                    break_op_found = true;
                    // We found a valid breakpoint opcode at this address, now restore
                    // the saved opcode.
                    if (m_task.WriteMemory(addr, bp->GetSavedOpcodeBytes(), break_op_size) == break_op_size)
                    {
                        verify = true;
                    }
                    else
                    {
                        GetError().SetErrorString("Memory write failed when restoring original opcode.");
                    }
                }
                else
                {
                    GetError().SetErrorString("Original breakpoint trap is no longer in memory.");
                    // Set verify to true and so we can check if the original opcode has already been restored
                    verify = true;
                }
            }
            else
            {
                if (log)
                    log->Printf ("ProcessMacOSXRemote::DisableBreakpoint (breakID = %d) is already disabled", breakID);
                // Set verify to true and so we can check if the original opcode is there
                verify = true;
            }

            if (verify)
            {
                uint8_t verify_opcode[break_op_size];
                // Verify that our original opcode made it back to the inferior
                if (m_task.ReadMemory(addr, verify_opcode, break_op_size) == break_op_size)
                {
                    // compare the memory we just read with the original opcode
                    if (memcmp(bp->GetSavedOpcodeBytes(), verify_opcode, break_op_size) == 0)
                    {
                        // SUCCESS
                        bp->SetEnabled(false);
                        if (log)
                            log->Printf ("ProcessMacOSXRemote::DisableBreakpoint (breakID = %d) SUCCESS", breakID);
                        return true;
                    }
                    else
                    {
                        if (break_op_found)
                            GetError().SetErrorString("Failed to restore original opcode.");
                    }
                }
                else
                {
                    GetError().SetErrorString("Failed to read memory to verify that breakpoint trap was restored.");
                }
            }
        }
        else
        {
            GetError().SetErrorString("Unable to read memory that should contain the breakpoint trap.");
        }
    }

    GetError().SetErrorToGenericError();
    return false;
}

bool
ProcessMacOSXRemote::EnableWatchpoint (WatchpointLocation *wp)
{
    if (wp)
    {
        lldb::user_id_t watchID = wp->GetID();
        lldb::addr_t addr = wp->GetAddress();
        Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_WATCHPOINTS);
        if (log)
            log->Printf ("ProcessMacOSXRemote::EnableWatchpoint(watchID = %d)", watchID);
        if (wp->IsEnabled())
        {
            if (log)
                log->Printf("ProcessMacOSXRemote::EnableWatchpoint(watchID = %d) addr = 0x%8.8llx: watchpoint already enabled.", watchID, (uint64_t)addr);
            return true;
        }
        else
        {
            ThreadMacOSX *thread = (ThreadMacOSX *)m_thread_list.FindThreadByID(wp->GetThreadID()).get();
            if (thread)
            {
                wp->SetHardwareIndex (thread->EnableHardwareWatchpoint (wp));
                if (wp->IsHardware ())
                {
                    wp->SetEnabled(true);
                    return true;
                }
            }
            else
            {
                GetError().SetErrorString("Watchpoints currently only support thread specific watchpoints.");
            }
        }
    }
    return false;
}

bool
ProcessMacOSXRemote::DisableWatchpoint (WatchpointLocation *wp)
{
    if (wp)
    {
        lldb::user_id_t watchID = wp->GetID();

        Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_WATCHPOINTS);

        lldb::addr_t addr = wp->GetAddress();
        if (log)
            log->Printf ("ProcessMacOSXRemote::DisableWatchpoint (watchID = %d) addr = 0x%8.8llx", watchID, (uint64_t)addr);

        if (wp->IsHardware())
        {
            ThreadMacOSX *thread = (ThreadMacOSX *)m_thread_list.FindThreadByID(wp->GetThreadID()).get();
            if (thread)
            {
                if (thread->DisableHardwareWatchpoint (wp))
                {
                    wp->SetEnabled(false);
                    if (log)
                        log->Printf ("ProcessMacOSXRemote::DisableWatchpoint (watchID = %d) addr = 0x%8.8llx (hardware) => success", watchID, (uint64_t)addr);
                    return true;
                }
            }
        }
        // TODO: clear software watchpoints if we implement them
    }
    else
    {
        GetError().SetErrorString("Watchpoint location argument was NULL.");
    }
    GetError().SetErrorToGenericError();
    return false;
}


static ProcessMacOSXRemote::CreateArchCalback
ArchDCScriptInterpreter::TypeMap(const ArchSpec& arch_spec, ProcessMacOSXRemote::CreateArchCalback callback, bool add )
{
    // We must wrap the "g_arch_map" file static in a function to avoid
    // any global constructors so we don't get a build verification error
    typedef std::multimap<ArchSpec, ProcessMacOSXRemote::CreateArchCalback> ArchToProtocolMap;
    static ArchToProtocolMap g_arch_map;

    if (add)
    {
        g_arch_map.insert(std::make_pair(arch_spec, callback));
        return callback;
    }
    else
    {
        ArchToProtocolMap::const_iterator pos = g_arch_map.find(arch_spec);
        if (pos != g_arch_map.end())
        {
            return pos->second;
        }
    }
    return NULL;
}

void
ProcessMacOSXRemote::AddArchCreateDCScriptInterpreter::Type(const ArchSpec& arch_spec, CreateArchCalback callback)
{
    ArchDCScriptInterpreter::TypeMap (arch_spec, callback, true);
}

ProcessMacOSXRemote::CreateArchCalback
ProcessMacOSXRemote::GetArchCreateDCScriptInterpreter::Type()
{
    return ArchDCScriptInterpreter::TypeMap (m_arch_spec, NULL, false);
}

void
ProcessMacOSXRemote::Clear()
{
    // Clear any cached thread list while the pid and task are still valid

    m_task.Clear();
    // Now clear out all member variables
    CloseChildFileDescriptors();

    m_flags = eFlagsNone;
    m_thread_list.Clear();
    {
        Mutex::Locker locker(m_exception_messages_mutex);
        m_exception_messages.clear();
    }

}


bool
ProcessMacOSXRemote::Kill (int signal)
{
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_PROCESS);
    if (log)
        log->Printf ("ProcessMacOSXRemote::Kill(signal = %d)", signal);
    State state = GetState();

    if (IsRunning(state))
    {
        if (::kill (GetID(), signal) == 0)
        {
            GetError().Clear();
        }
        else
        {
            GetError().SetErrorToErrno();
            GetError().LogIfError(log, "ProcessMacOSXRemote::Kill(%d)", signal);
        }
    }
    else
    {
        if (log)
            log->Printf ("ProcessMacOSXRemote::Kill(signal = %d) pid %u (task = 0x%4.4x) was't running, ignoring...", signal, GetID(), m_task.TaskPort());
        GetError().Clear();
    }
    return GetError().Success();

}


bool
ProcessMacOSXRemote::Detach()
{
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSXRemote::Detach()");

    State state = GetState();

    if (!IsRunning(state))
    {
        // Resume our process
        PrivateResume(LLDB_INVALID_THREAD_ID);

        // We have resumed and now we wait for that event to get posted
        Event event;
        if (WaitForPrivateEvents(LLDB_EVENT_RUNNING, &event, 2) == false)
            return false;


        // We need to be stopped in order to be able to detach, so we need
        // to send ourselves a SIGSTOP
        if (Kill(SIGSTOP))
        {
            Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_PROCESS);

            lldb::pid_t pid = GetID();
            // Wait for our process stop event to get posted
            if (WaitForPrivateEvents(LLDB_EVENT_STOPPED, &event, 2) == false)
            {
                GetError().Log(log, "::kill (pid = %u, SIGSTOP)", pid);
                return false;
            }

            // Shut down the exception thread and cleanup our exception remappings
            m_task.ShutDownExceptionThread();

            // Detach from our process while we are stopped.
            errno = 0;

            // Detach from our process
            ::ptrace (PT_DETACH, pid, (caddr_t)1, 0);

            GetError().SetErrorToErrno();

            if (log || GetError().Fail())
                GetError().Log(log, "::ptrace (PT_DETACH, %u, (caddr_t)1, 0)", pid);

            // Resume our task
            m_task.Resume();

            // NULL our task out as we have already retored all exception ports
            m_task.Clear();

            // Clear out any notion of the process we once were
            Clear();
        }
    }
    else
    {
        ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSXRemote::Detach() error: process must be stopped (SIGINT the process first).");
    }
    return false;
}



void
ProcessMacOSXRemote::ReplyToAllExceptions()
{
    Mutex::Locker locker(m_exception_messages_mutex);
    if (m_exception_messages.empty() == false)
    {
        Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_EXCEPTIONS);

        MachException::Message::iterator pos;
        MachException::Message::iterator begin = m_exception_messages.begin();
        MachException::Message::iterator end = m_exception_messages.end();
        for (pos = begin; pos != end; ++pos)
        {
            if (log)
                log->Printf ("Replying to exception %d...", std::distance(begin, pos));
            int resume_signal = 0;
            ThreadSP thread_sp = m_thread_list.FindThreadByID(pos->state.thread_port);
            if (thread_sp.get())
                resume_signal = thread_sp->GetResumeSignal();
            GetError() = pos->Reply (Task().TaskPort(), GetID(), resume_signal);
            GetError().LogIfError(log, "Error replying to exception");
        }

        // Erase all exception message as we should have used and replied
        // to them all already.
        m_exception_messages.clear();
    }
}
void
ProcessMacOSXRemote::PrivateResume (lldb::tid_t tid)
{
    Mutex::Locker locker(m_exception_messages_mutex);
    ReplyToAllExceptions();

    // Let the thread prepare to resume and see if any threads want us to
    // step over a breakpoint instruction (ProcessWillResume will modify
    // the value of stepOverBreakInstruction).
    //StateType process_state = m_thread_list.ProcessWillResume(this);

    // Set our state accordingly
    SetState(eStateRunning);

    // Now resume our task.
    GetError() = m_task.Resume();

}

// Called by the exception thread when an exception has been received from
// our process. The exception message is completely filled and the exception
// data has already been copied.
void
ProcessMacOSXRemote::ExceptionMessageReceived (const MachException::Message& exceptionMessage)
{
    Mutex::Locker locker(m_exception_messages_mutex);

    if (m_exception_messages.empty())
        m_task.Suspend();

    ProcessMacOSXLog::LogIf (PD_LOG_EXCEPTIONS, "ProcessMacOSXRemote::ExceptionMessageReceived ( )");

    // Use a locker to automatically unlock our mutex in case of exceptions
    // Add the exception to our internal exception stack
    m_exception_messages.push_back(exceptionMessage);
}


//bool
//ProcessMacOSXRemote::GetProcessInfo (struct kinfo_proc* proc_info)
//{
//  int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, GetID() };
//  size_t buf_size = sizeof(struct kinfo_proc);
//
//  if (::sysctl (mib, (unsigned)(sizeof(mib)/sizeof(int)), &proc_info, &buf_size, NULL, 0) == 0)
//      return buf_size > 0;
//
//  return false;
//}
//
//
void
ProcessMacOSXRemote::ExceptionMessageBundleComplete()
{
    // We have a complete bundle of exceptions for our child process.
    Mutex::Locker locker(m_exception_messages_mutex);
    ProcessMacOSXLog::LogIf (PD_LOG_EXCEPTIONS, "%s: %d exception messages.", __PRETTY_FUNCTION__, m_exception_messages.size());
    if (!m_exception_messages.empty())
    {
        SetState (eStateStopped);
    }
    else
    {
        ProcessMacOSXLog::LogIf (PD_LOG_EXCEPTIONS, "%s empty exception messages bundle.", __PRETTY_FUNCTION__, m_exception_messages.size());
    }
}

bool
ProcessMacOSXRemote::ReleaseChildFileDescriptors ( int *stdin_fileno, int *stdout_fileno, int *stderr_fileno )
{
    if (stdin_fileno)
        *stdin_fileno = m_child_stdin;
    if (stdout_fileno)
        *stdout_fileno = m_child_stdout;
    if (stderr_fileno)
        *stderr_fileno = m_child_stderr;
    // Stop the stdio thread if we have one, but don't have it close the child
    // file descriptors since we are giving control of these descriptors to the
    // caller
    bool close_child_fds = false;
    StopSTDIOThread(close_child_fds);
    return true;
}

void
ProcessMacOSXRemote::AppendSTDOUT (char* s, size_t len)
{
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSXRemote::%s (<%d> %s) ...", __FUNCTION__, len, s);
    Mutex::Locker locker(m_stdio_mutex);
    m_stdout_data.append(s, len);
    AppendEvent (LLDB_EVENT_STDIO);
}

void *
ProcessMacOSXRemote::STDIOThread(void *arg)
{
    ProcessMacOSXRemote *proc = (ProcessMacOSXRemote*) arg;

    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_PROCESS);
    if (log)
        log->Printf ("ProcessMacOSXRemote::%s (arg = %p) thread starting...", __FUNCTION__, arg);

    // We start use a base and more options so we can control if we
    // are currently using a timeout on the mach_msg. We do this to get a
    // bunch of related exceptions on our exception port so we can process
    // then together. When we have multiple threads, we can get an exception
    // per thread and they will come in consecutively. The main thread loop
    // will start by calling mach_msg to without having the MACH_RCV_TIMEOUT
    // flag set in the options, so we will wait forever for an exception on
    // our exception port. After we get one exception, we then will use the
    // MACH_RCV_TIMEOUT option with a zero timeout to grab all other current
    // exceptions for our process. After we have received the last pending
    // exception, we will get a timeout which enables us to then notify
    // our main thread that we have an exception bundle available. We then wait
    // for the main thread to tell this exception thread to start trying to get
    // exceptions messages again and we start again with a mach_msg read with
    // infinite timeout.
    Error err;
    int stdout_fd = proc->GetStdoutFileDescriptor();
    int stderr_fd = proc->GetStderrFileDescriptor();
    if (stdout_fd == stderr_fd)
        stderr_fd = -1;

    while (stdout_fd >= 0 || stderr_fd >= 0)
    {
        ::pthread_testcancel ();

        fd_set read_fds;
        FD_ZERO (&read_fds);
        if (stdout_fd >= 0)
            FD_SET (stdout_fd, &read_fds);
        if (stderr_fd >= 0)
            FD_SET (stderr_fd, &read_fds);
        int nfds = std::max<int>(stdout_fd, stderr_fd) + 1;

        int num_set_fds = select (nfds, &read_fds, NULL, NULL, NULL);
        if (log)
            log->Printf("select (nfds, &read_fds, NULL, NULL, NULL) => %d", num_set_fds);

        if (num_set_fds < 0)
        {
            int select_errno = errno;
            if (log)
            {
                err.SetError (select_errno, Error::POSIX);
                err.LogIfError(log, "select (nfds, &read_fds, NULL, NULL, NULL) => %d", num_set_fds);
            }

            switch (select_errno)
            {
            case EAGAIN:    // The kernel was (perhaps temporarily) unable to allocate the requested number of file descriptors, or we have non-blocking IO
                break;
            case EBADF:     // One of the descriptor sets specified an invalid descriptor.
                return NULL;
                break;
            case EINTR:     // A signal was delivered before the time limit expired and before any of the selected events occurred.
            case EINVAL:    // The specified time limit is invalid. One of its components is negative or too large.
            default:        // Other unknown error
                break;
            }
        }
        else if (num_set_fds == 0)
        {
        }
        else
        {
            char s[1024];
            s[sizeof(s)-1] = '\0';  // Ensure we have NULL termination
            int bytes_read = 0;
            if (stdout_fd >= 0 && FD_ISSET (stdout_fd, &read_fds))
            {
                do
                {
                    bytes_read = ::read (stdout_fd, s, sizeof(s)-1);
                    if (bytes_read < 0)
                    {
                        int read_errno = errno;
                        if (log)
                            log->Printf("read (stdout_fd, ) => %d   errno: %d (%s)", bytes_read, read_errno, strerror(read_errno));
                    }
                    else if (bytes_read == 0)
                    {
                        // EOF...
                        if (log)
                            log->Printf("read (stdout_fd, ) => %d  (reached EOF for child STDOUT)", bytes_read);
                        stdout_fd = -1;
                    }
                    else if (bytes_read > 0)
                    {
                        proc->AppendSTDOUT(s, bytes_read);
                    }

                } while (bytes_read > 0);
            }

            if (stderr_fd >= 0 && FD_ISSET (stderr_fd, &read_fds))
            {
                do
                {
                    bytes_read = ::read (stderr_fd, s, sizeof(s)-1);
                    if (bytes_read < 0)
                    {
                        int read_errno = errno;
                        if (log)
                            log->Printf("read (stderr_fd, ) => %d   errno: %d (%s)", bytes_read, read_errno, strerror(read_errno));
                    }
                    else if (bytes_read == 0)
                    {
                        // EOF...
                        if (log)
                            log->Printf("read (stderr_fd, ) => %d  (reached EOF for child STDERR)", bytes_read);
                        stderr_fd = -1;
                    }
                    else if (bytes_read > 0)
                    {
                        proc->AppendSTDOUT(s, bytes_read);
                    }

                } while (bytes_read > 0);
            }
        }
    }

    if (log)
        log->Printf("ProcessMacOSXRemote::%s (%p): thread exiting...", __FUNCTION__, arg);

    return NULL;
}

lldb::pid_t
ProcessMacOSXRemote::AttachForDebug (lldb::pid_t pid)
{
    // Clear out and clean up from any current state
    Clear();
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_PROCESS);
    if (pid != 0)
    {
        SetState(eStateAttaching);
        SetID(pid);
        // Let ourselves know we are going to be using SBS if the correct flag bit is set...
#if defined (__arm__)
        if (IsSBProcess(pid))
            m_flags |= eFlagsUsingSBS;
#endif
        m_task.StartExceptionThread(GetError());

        if (GetError().Success())
        {
            if (ptrace (PT_ATTACHEXC, pid, 0, 0) == 0)
            {
                m_flags.Set (eFlagsAttached);
                // Sleep a bit to let the exception get received and set our process status
                // to stopped.
                ::usleep(250000);
                if (log)
                    log->Printf ("successfully attached to pid %d", pid);
                return GetID();
            }
            else
            {
                GetError().SetErrorToErrno();
                if (log)
                    log->Printf ("error: failed to attach to pid %d", pid);
            }
        }
        else
        {
            GetError().Log(log, "ProcessMacOSXRemote::%s (pid = %i) failed to start exception thread", __FUNCTION__, pid);
        }
    }
    return LLDB_INVALID_PROCESS_ID;
}

lldb::pid_t
ProcessMacOSXRemote::LaunchForDebug
(
    const char *path,
    char const *argv[],
    char const *envp[],
    ArchSpec& arch_spec,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    PDLaunchType launch_type,
    Error &launch_err)
{
    // Clear out and clean up from any current state
    Clear();

    m_arch_spec = arch_spec;

    if (launch_type == eLaunchDefault)
        launch_type = eLaunchPosixSpawn;

    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_PROCESS);
    if (log)
        log->Printf ("%s( path = '%s', argv = %p, envp = %p, launch_type = %u )", __FUNCTION__, path, argv, envp, launch_type);

    // Fork a child process for debugging
    SetState(eStateLaunching);
    switch (launch_type)
    {
    case eLaunchForkExec:
        SetID(ProcessMacOSXRemote::ForkChildForPTraceDebugging(path, argv, envp, arch_spec, stdin_path, stdout_path, stderr_path, this, launch_err));
        break;

    case eLaunchPosixSpawn:
        SetID(ProcessMacOSXRemote::PosixSpawnChildForPTraceDebugging(path, argv, envp, arch_spec, stdin_path, stdout_path, stderr_path, this, launch_err));
        break;

#if defined (__arm__)

    case eLaunchSpringBoard:
        {
            const char *app_ext = strstr(path, ".app");
            if (app_ext != NULL)
            {
                std::string app_bundle_path(path, app_ext + strlen(".app"));
                return SBLaunchForDebug (app_bundle_path.c_str(), argv, envp, arch_spec, stdin_path, stdout_path, stderr_path, launch_err);
            }
        }
        break;

#endif

    default:
        // Invalid  launch
        launch_err.SetErrorToGenericError ();
        return LLDB_INVALID_PROCESS_ID;
    }

    lldb::pid_t pid = GetID();

    if (pid == LLDB_INVALID_PROCESS_ID)
    {
        // If we don't have a valid process ID and no one has set the error,
        // then return a generic error
        if (launch_err.Success())
            launch_err.SetErrorToGenericError ();
    }
    else
    {
        // Make sure we can get our task port before going any further
        m_task.TaskPortForProcessID (launch_err);

        // If that goes well then kick off our exception thread
        if (launch_err.Success())
            m_task.StartExceptionThread(launch_err);

        if (launch_err.Success())
        {
            //m_path = path;
//          size_t i;
//          if (argv)
//          {
//              char const *arg;
//              for (i=0; (arg = argv[i]) != NULL; i++)
//                  m_args.push_back(arg);
//          }

            StartSTDIOThread();

            if (launch_type == eLaunchPosixSpawn)
            {

                //SetState (eStateAttaching);
                errno = 0;
                if (::ptrace (PT_ATTACHEXC, pid, 0, 0) == 0)
                    launch_err.Clear();
                else
                    launch_err.SetErrorToErrno();

                if (launch_err.Fail() || log)
                    launch_err.Log(log, "::ptrace (PT_ATTACHEXC, pid = %i, 0, 0 )", pid);

                if (launch_err.Success())
                    m_flags.Set (eFlagsAttached);
                else
                    SetState (eStateExited);
            }
            else
            {
                launch_err.Clear();
            }
        }
        else
        {
            // We were able to launch the process, but not get its task port
            // so now we need to make it sleep with da fishes.
            SetID(LLDB_INVALID_PROCESS_ID);
            ::kill (pid, SIGCONT);
            ::kill (pid, SIGKILL);
            pid = LLDB_INVALID_PROCESS_ID;
        }

    }
    return pid;
}

lldb::pid_t
ProcessMacOSXRemote::PosixSpawnChildForPTraceDebugging
(
    const char *path,
    char const *argv[],
    char const *envp[],
    ArchSpec& arch_spec,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    ProcessMacOSXRemote* process,
    Error &err
)
{
    posix_spawnattr_t attr;

    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_PROCESS);

    Error local_err;    // Errors that don't affect the spawning.
    if (log)
        log->Printf ("%s ( path='%s', argv=%p, envp=%p, process )", __FUNCTION__, path, argv, envp);
    err.SetError( ::posix_spawnattr_init (&attr), Error::POSIX);
    if (err.Fail() || log)
        err.Log(log, "::posix_spawnattr_init ( &attr )");
    if (err.Fail())
        return LLDB_INVALID_PROCESS_ID;

    err.SetError( ::posix_spawnattr_setflags (&attr, POSIX_SPAWN_START_SUSPENDED), Error::POSIX);
    if (err.Fail() || log)
        err.Log(log, "::posix_spawnattr_setflags ( &attr, POSIX_SPAWN_START_SUSPENDED )");
    if (err.Fail())
        return LLDB_INVALID_PROCESS_ID;

#if !defined(__arm__)

    // We don't need to do this for ARM, and we really shouldn't now that we
    // have multiple CPU subtypes and no posix_spawnattr call that allows us
    // to set which CPU subtype to launch...
    cpu_type_t cpu = arch_spec.GetCPUType();
    if (cpu != 0 && cpu != CPU_TYPE_ANY && cpu != LLDB_INVALID_CPUTYPE)
    {
        size_t ocount = 0;
        err.SetError( ::posix_spawnattr_setbinpref_np (&attr, 1, &cpu, &ocount), Error::POSIX);
        if (err.Fail() || log)
            err.Log(log, "::posix_spawnattr_setbinpref_np ( &attr, 1, cpu_type = 0x%8.8x, count => %zu )", cpu, ocount);

        if (err.Fail() != 0 || ocount != 1)
            return LLDB_INVALID_PROCESS_ID;
    }

#endif

    PseudoTerminal pty;

    posix_spawn_file_actions_t file_actions;
    err.SetError( ::posix_spawn_file_actions_init (&file_actions), Error::POSIX);
    int file_actions_valid = err.Success();
    if (!file_actions_valid || log)
        err.Log(log, "::posix_spawn_file_actions_init ( &file_actions )");
    Error stdio_err;
    lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
    if (file_actions_valid)
    {
        // If the user specified any STDIO files, then use those
        if (stdin_path || stdout_path || stderr_path)
        {
            process->SetSTDIOIsOurs(false);
            if (stderr_path != NULL && stderr_path[0])
            {
                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDERR_FILENO,    stderr_path, O_RDWR, 0), Error::POSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.Log(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDERR_FILENO, path = '%s', oflag = O_RDWR, mode = 0 )", stderr_path);
            }

            if (stdin_path != NULL && stdin_path[0])
            {
                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDIN_FILENO, stdin_path, O_RDONLY, 0), Error::POSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.Log(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDIN_FILENO, path = '%s', oflag = O_RDONLY, mode = 0 )", stdin_path);
            }

            if (stdout_path != NULL && stdout_path[0])
            {
                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDOUT_FILENO,    stdout_path, O_WRONLY, 0), Error::POSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.Log(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDOUT_FILENO, path = '%s', oflag = O_WRONLY, mode = 0 )", stdout_path);
            }
        }
        else
        {
            // The user did not specify any STDIO files, use a pseudo terminal.
            // Callers can then access the file handles using the
            // ProcessMacOSXRemote::ReleaseChildFileDescriptors() function, otherwise
            // this class will spawn a thread that tracks STDIO and buffers it.
            process->SetSTDIOIsOurs(true);
            if (pty.OpenFirstAvailableMaster(O_RDWR, &stdio_err))
            {
                const char* slave_name = pty.GetSlaveName(&stdio_err);
                if (slave_name == NULL)
                    slave_name = "/dev/null";
                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDERR_FILENO,    slave_name, O_RDWR, 0), Error::POSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.Log(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDERR_FILENO, path = '%s', oflag = O_RDWR, mode = 0 )", slave_name);

                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDIN_FILENO, slave_name, O_RDONLY, 0), Error::POSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.Log(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDIN_FILENO, path = '%s', oflag = O_RDONLY, mode = 0 )", slave_name);

                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDOUT_FILENO,    slave_name, O_WRONLY, 0), Error::POSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.Log(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDOUT_FILENO, path = '%s', oflag = O_WRONLY, mode = 0 )", slave_name);
            }
        }
        err.SetError( ::posix_spawnp (&pid, path, &file_actions, &attr, (char * const*)argv, (char * const*)envp), Error::POSIX);
        if (err.Fail() || log)
            err.Log(log, "::posix_spawnp ( pid => %i, path = '%s', file_actions = %p, attr = %p, argv = %p, envp = %p )", pid, path, &file_actions, &attr, argv, envp);

        if (stdio_err.Success())
        {
            // If we have a valid process and we created the STDIO file handles,
            // then remember them on our process class so we can spawn a STDIO
            // thread and close them when we are done with them.
            if (process != NULL && process->STDIOIsOurs())
            {
                int master_fd = pty.ReleaseMasterFileDescriptor ();
                process->SetChildFileDescriptors (master_fd, master_fd, master_fd);
            }
        }
    }
    else
    {
        err.SetError( ::posix_spawnp (&pid, path, NULL, &attr, (char * const*)argv, (char * const*)envp), Error::POSIX);
        if (err.Fail() || log)
            err.Log(log, "::posix_spawnp ( pid => %i, path = '%s', file_actions = %p, attr = %p, argv = %p, envp = %p )", pid, path, NULL, &attr, argv, envp);
    }

    // We have seen some cases where posix_spawnp was returning a valid
    // looking pid even when an error was returned, so clear it out
    if (err.Fail())
        pid = LLDB_INVALID_PROCESS_ID;

    if (file_actions_valid)
    {
        local_err.SetError( ::posix_spawn_file_actions_destroy (&file_actions), Error::POSIX);
        if (local_err.Fail() || log)
            local_err.Log(log, "::posix_spawn_file_actions_destroy ( &file_actions )");
    }

    return pid;
}

lldb::pid_t
ProcessMacOSXRemote::ForkChildForPTraceDebugging
(
    const char *path,
    char const *argv[],
    char const *envp[],
    ArchSpec& arch_spec,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    ProcessMacOSXRemote* process,
    Error &launch_err
)
{
    lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;

    if (stdin_path || stdout_path || stderr_path)
    {
        assert(!"TODO: ForkChildForPTraceDebugging doesn't currently support fork/exec with user file handles...");
    }
    else
    {

        // Use a fork that ties the child process's stdin/out/err to a pseudo
        // terminal so we can read it in our ProcessMacOSXRemote::STDIOThread
        // as unbuffered io.
        PseudoTerminal pty;
        pid = pty.Fork(&launch_err);

        if (pid < 0)
        {
            //--------------------------------------------------------------
            // Error during fork.
            //--------------------------------------------------------------
            return pid;
        }
        else if (pid == 0)
        {
            //--------------------------------------------------------------
            // Child process
            //--------------------------------------------------------------
            ::ptrace (PT_TRACE_ME, 0, 0, 0);    // Debug this process
            ::ptrace (PT_SIGEXC, 0, 0, 0);    // Get BSD signals as mach exceptions

            // If our parent is setgid, lets make sure we don't inherit those
            // extra powers due to nepotism.
            ::setgid (getgid ());

            // Let the child have its own process group. We need to execute
            // this call in both the child and parent to avoid a race condition
            // between the two processes.
            ::setpgid (0, 0);    // Set the child process group to match its pid

            // Sleep a bit to before the exec call
            ::sleep (1);

            // Turn this process into
            ::execv (path, (char * const *)argv);
            // Exit with error code. Child process should have taken
            // over in above exec call and if the exec fails it will
            // exit the child process below.
            ::exit (127);
        }
        else
        {
            //--------------------------------------------------------------
            // Parent process
            //--------------------------------------------------------------
            // Let the child have its own process group. We need to execute
            // this call in both the child and parent to avoid a race condition
            // between the two processes.
            ::setpgid (pid, pid);    // Set the child process group to match its pid

            if (process != NULL)
            {
                // Release our master pty file descriptor so the pty class doesn't
                // close it and so we can continue to use it in our STDIO thread
                int master_fd = pty.ReleaseMasterFileDescriptor ();
                process->SetChildFileDescriptors (master_fd, master_fd, master_fd);
            }
        }
    }
    return pid;
}

#if defined (__arm__)

lldb::pid_t
ProcessMacOSXRemote::SBLaunchForDebug
(
    const char *path,
    char const *argv[],
    char const *envp[],
    ArchSpec& arch_spec,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    Error &launch_err
)
{
    // Clear out and clean up from any current state
    Clear();

    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s( '%s', argv)", __FUNCTION__, path);

    // Fork a child process for debugging
    SetState(eStateLaunching);
    m_pid = ProcessMacOSXRemote::SBLaunchForDebug(path, argv, envp, this, launch_err);
    if (m_pid != 0)
    {
        m_flags |= eFlagsUsingSBS;
        //m_path = path;
//        size_t i;
//        char const *arg;
//        for (i=0; (arg = argv[i]) != NULL; i++)
//            m_args.push_back(arg);
        m_task.StartExceptionThread();
        StartSTDIOThread();
        SetState (eStateAttaching);
        int err = ptrace (PT_ATTACHEXC, m_pid, 0, 0);
        if (err == 0)
        {
            m_flags |= eFlagsAttached;
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "successfully attached to pid %d", m_pid);
        }
        else
        {
            SetState (eStateExited);
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "error: failed to attach to pid %d", m_pid);
        }
    }
    return m_pid;
}

#include <servers/bootstrap.h>
#include "CFBundle.h"
#include "CFData.h"
#include "CFString.h"

lldb::pid_t
ProcessMacOSXRemote::SBLaunchForDebug
(
    const char *app_bundle_path,
    char const *argv[],
    char const *envp[],
    ArchSpec& arch_spec,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    ProcessMacOSXRemote* process,
    Error &launch_err
)
{
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s( '%s', argv, %p)", __FUNCTION__, app_bundle_path, process);
    CFAllocatorRef alloc = kCFAllocatorDefault;
    if (argv[0] == NULL)
        return LLDB_INVALID_PROCESS_ID;

    size_t argc = 0;
    // Count the number of arguments
    while (argv[argc] != NULL)
        argc++;

    // Enumerate the arguments
    size_t first_launch_arg_idx = 1;
    CFReleaser<CFMutableArrayRef> launch_argv;

    if (argv[first_launch_arg_idx])
    {
        size_t launch_argc = argc > 0 ? argc - 1 : 0;
        launch_argv.reset (::CFArrayCreateMutable (alloc, launch_argc, &kCFTypeArrayCallBacks));
        size_t i;
        char const *arg;
        CFString launch_arg;
        for (i=first_launch_arg_idx; (i < argc) && ((arg = argv[i]) != NULL); i++)
        {
            launch_arg.reset(::CFStringCreateWithCString (alloc, arg, kCFStringEncodingUTF8));
            if (launch_arg.get() != NULL)
                CFArrayAppendValue(launch_argv.get(), launch_arg.get());
            else
                break;
        }
    }

    // Next fill in the arguments dictionary.  Note, the envp array is of the form
    // Variable=value but SpringBoard wants a CF dictionary.  So we have to convert
    // this here.

    CFReleaser<CFMutableDictionaryRef> launch_envp;

    if (envp[0])
    {
        launch_envp.reset(::CFDictionaryCreateMutable(alloc, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
        const char *value;
        int name_len;
        CFString name_string, value_string;

        for (int i = 0; envp[i] != NULL; i++)
        {
            value = strstr (envp[i], "=");

            // If the name field is empty or there's no =, skip it.  Somebody's messing with us.
            if (value == NULL || value == envp[i])
                continue;

            name_len = value - envp[i];

            // Now move value over the "="
            value++;

            name_string.reset(::CFStringCreateWithBytes(alloc, (const UInt8 *) envp[i], name_len, kCFStringEncodingUTF8, false));
            value_string.reset(::CFStringCreateWithCString(alloc, value, kCFStringEncodingUTF8));
            CFDictionarySetValue (launch_envp.get(), name_string.get(), value_string.get());
        }
    }

    CFString stdout_cf_path;
    CFString stderr_cf_path;
    PseudoTerminal pty;

    if (stdin_path || stdout_path || stderr_path)
    {
        process->SetSTDIOIsOurs(false);
        if (stdout_path)
            stdout_cf_path.SetFileSystemRepresentation (stdout_path);
        if (stderr_path)
            stderr_cf_path.SetFileSystemRepresentation (stderr_path);
    }
    else
    {
        process->SetSTDIOIsOurs(true);
        PseudoTerminal::Error pty_err = pty.OpenFirstAvailableMaster(O_RDWR);
        if (pty_err == PseudoTerminal::success)
        {
            const char* slave_name = pty.SlaveName();
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() successfully opened master pty, slave is %s", __FUNCTION__, slave_name);
            if (slave_name && slave_name[0])
            {
                ::chmod (slave_name, S_IRWXU | S_IRWXG | S_IRWXO);
                stdout_cf_path.SetFileSystemRepresentation (slave_name);
                stderr_cf_path.(stdout_cf_path);
            }
        }
    }

    if (stdout_cf_path.get() == NULL)
        stdout_cf_path.SetFileSystemRepresentation ("/dev/null");
    if (stderr_cf_path.get() == NULL)
        stderr_cf_path.SetFileSystemRepresentation ("/dev/null");

    CFBundle bundle(app_bundle_path);
    CFStringRef bundleIDCFStr = bundle.GetIdentifier();
    std::string bundleID;
    if (CFString::UTF8(bundleIDCFStr, bundleID) == NULL)
    {
        struct stat app_bundle_stat;
        if (::stat (app_bundle_path, &app_bundle_stat) < 0)
        {
            launch_err.SetError(errno, Error::POSIX);
            launch_err.SetErrorStringWithFormat ("%s: \"%s\".\n", launch_err.AsString(), app_bundle_path);
        }
        else
        {
            launch_err.SetError(-1, Error::Generic);
            launch_err.SetErrorStringWithFormat ("Failed to extract CFBundleIdentifier from %s.\n", app_bundle_path);
        }
        return LLDB_INVALID_PROCESS_ID;
    }
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() extracted CFBundleIdentifier: %s", __FUNCTION__, bundleID.c_str());


    CFData argv_data(NULL);

    if (launch_argv.get())
    {
        if (argv_data.Serialize(launch_argv.get(), kCFPropertyListBinaryFormat_v1_0) == NULL)
        {
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() error: failed to serialize launch arg array...", __FUNCTION__);
            return LLDB_INVALID_PROCESS_ID;
        }
    }

    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() serialized launch arg array", __FUNCTION__);

    // Find SpringBoard
    SBSApplicationLaunchError sbs_error = 0;
    sbs_error = SBSLaunchApplication (  bundleIDCFStr,
                                        (CFURLRef)NULL,         // openURL
                                        launch_argv.get(),
                                        launch_envp.get(),      // CFDictionaryRef environment
                                        stdout_cf_path.get(),
                                        stderr_cf_path.get(),
                                        SBSApplicationLaunchWaitForDebugger | SBSApplicationLaunchUnlockDevice);


    launch_err.SetError(sbs_error, Error::SpringBoard);

    if (sbs_error == SBSApplicationLaunchErrorSuccess)
    {
        static const useconds_t pid_poll_interval = 200000;
        static const useconds_t pid_poll_timeout = 30000000;

        useconds_t pid_poll_total = 0;

        lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
        Boolean pid_found = SBSProcessIDForDisplayIdentifier(bundleIDCFStr, &pid);
        // Poll until the process is running, as long as we are getting valid responses and the timeout hasn't expired
        // A return PID of 0 means the process is not running, which may be because it hasn't been (asynchronously) started
        // yet, or that it died very quickly (if you weren't using waitForDebugger).
        while (!pid_found && pid_poll_total < pid_poll_timeout)
        {
            usleep (pid_poll_interval);
            pid_poll_total += pid_poll_interval;
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() polling Springboard for pid for %s...", __FUNCTION__, bundleID.c_str());
            pid_found = SBSProcessIDForDisplayIdentifier(bundleIDCFStr, &pid);
        }

        if (pid_found)
        {
            // If we have a valid process and we created the STDIO file handles,
            // then remember them on our process class so we can spawn a STDIO
            // thread and close them when we are done with them.
            if (process != NULL && process->STDIOIsOurs())
            {
                // Release our master pty file descriptor so the pty class doesn't
                // close it and so we can continue to use it in our STDIO thread
                int master_fd = pty.ReleaseMasterFD();
                process->SetChildFileDescriptors(master_fd, master_fd, master_fd);
            }
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() => pid = %4.4x", __FUNCTION__, pid);
        }
        else
        {
            LogError("failed to lookup the process ID for CFBundleIdentifier %s.", bundleID.c_str());
        }
        return pid;
    }

    LogError("unable to launch the application with CFBundleIdentifier '%s' sbs_error = %u", bundleID.c_str(), sbs_error);
    return LLDB_INVALID_PROCESS_ID;
}

#endif // #if defined (__arm__)

