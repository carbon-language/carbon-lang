//===-- MachTask.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MachTask.h"

// C Includes
// C++ Includes

// Other libraries and framework includes
#if defined (__arm__)

#include <CoreFoundation/CoreFoundation.h>
#include <SpringBoardServices/SpringBoardServer.h>
#include <SpringBoardServices/SBSWatchdogAssertion.h>

#endif

#include "lldb/Host/Host.h"
#include "lldb/Core/DataExtractor.h"

// Project includes
#include "ProcessMacOSX.h"
#include "ProcessMacOSXLog.h"


using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// MachTask constructor
//----------------------------------------------------------------------
MachTask::MachTask(ProcessMacOSX *process) :
    m_process (process),
    m_task (TASK_NULL),
    m_vm_memory (),
    m_exc_port_info(),
    m_exception_thread (0),
    m_exception_port (MACH_PORT_NULL)
{
    memset(&m_exc_port_info, 0, sizeof(m_exc_port_info));

}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
MachTask::~MachTask()
{
    Clear();
}


//----------------------------------------------------------------------
// MachTask::Suspend
//----------------------------------------------------------------------
kern_return_t
MachTask::Suspend()
{
    Error err;
    task_t task = GetTaskPort();
    err = ::task_suspend (task);
    LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_TASK));
    if (log || err.Fail())
        err.PutToLog(log.get(), "::task_suspend ( target_task = 0x%4.4x )", task);
    return err.GetError();
}


//----------------------------------------------------------------------
// MachTask::Resume
//----------------------------------------------------------------------
kern_return_t
MachTask::Resume()
{
    Error err;
    task_t task = GetTaskPort();
    err = ::task_resume (task);
    LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_TASK));
    if (log || err.Fail())
        err.PutToLog(log.get(), "::task_resume ( target_task = 0x%4.4x )", task);
    return err.GetError();
}

int32_t
MachTask::GetSuspendCount () const
{
    struct task_basic_info task_info;
    if (BasicInfo(&task_info) == KERN_SUCCESS)
        return task_info.suspend_count;
    return -1;
}

//----------------------------------------------------------------------
// MachTask::ExceptionPort
//----------------------------------------------------------------------
mach_port_t
MachTask::ExceptionPort() const
{
    return m_exception_port;
}

//----------------------------------------------------------------------
// MachTask::ExceptionPortIsValid
//----------------------------------------------------------------------
bool
MachTask::ExceptionPortIsValid() const
{
    return MACH_PORT_VALID(m_exception_port);
}


//----------------------------------------------------------------------
// MachTask::Clear
//----------------------------------------------------------------------
void
MachTask::Clear()
{
    // Do any cleanup needed for this task
    m_task = TASK_NULL;
    m_exception_thread = 0;
    m_exception_port = MACH_PORT_NULL;

}


//----------------------------------------------------------------------
// MachTask::SaveExceptionPortInfo
//----------------------------------------------------------------------
kern_return_t
MachTask::SaveExceptionPortInfo()
{
    return m_exc_port_info.Save(GetTaskPort());
}

//----------------------------------------------------------------------
// MachTask::RestoreExceptionPortInfo
//----------------------------------------------------------------------
kern_return_t
MachTask::RestoreExceptionPortInfo()
{
    return m_exc_port_info.Restore(GetTaskPort());
}


//----------------------------------------------------------------------
// MachTask::ReadMemory
//----------------------------------------------------------------------
size_t
MachTask::ReadMemory (lldb::addr_t addr, void *buf, size_t size, Error& error)
{
    size_t n = 0;
    task_t task = GetTaskPort();
    if (task != TASK_NULL)
    {
        n = m_vm_memory.Read(task, addr, buf, size, error);
        LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_MEMORY));
        if (log)
        {
            log->Printf ("MachTask::ReadMemory ( addr = 0x%16.16llx, size = %zu, buf = %8.8p) => %u bytes read", (uint64_t)addr, size, buf, n);
            if (log->GetMask().Test(PD_LOG_MEMORY_DATA_LONG) || (log->GetMask().Test(PD_LOG_MEMORY_DATA_SHORT) && size <= 8))
            {
                DataExtractor data((uint8_t*)buf, n, eByteOrderHost, 4);
                data.PutToLog(log.get(), 0, n, addr, 16, DataExtractor::TypeUInt8);
            }
        }
    }
    return n;
}


//----------------------------------------------------------------------
// MachTask::WriteMemory
//----------------------------------------------------------------------
size_t
MachTask::WriteMemory (lldb::addr_t addr, const void *buf, size_t size, Error& error)
{
    size_t n = 0;
    task_t task = GetTaskPort();
    if (task != TASK_NULL)
    {
        n = m_vm_memory.Write(task, addr, buf, size, error);
        LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_MEMORY));
        if (log)
        {
            log->Printf ("MachTask::WriteMemory ( addr = 0x%16.16llx, size = %zu, buf = %8.8p) => %u bytes written", (uint64_t)addr, size, buf, n);
            if (log->GetMask().Test(PD_LOG_MEMORY_DATA_LONG) || (log->GetMask().Test(PD_LOG_MEMORY_DATA_SHORT) && size <= 8))
            {
                DataExtractor data((uint8_t*)buf, n, eByteOrderHost, 4);
                data.PutToLog(log.get(), 0, n, addr, 16, DataExtractor::TypeUInt8);
            }
        }
    }
    return n;
}

//----------------------------------------------------------------------
// MachTask::AllocateMemory
//----------------------------------------------------------------------
lldb::addr_t
MachTask::AllocateMemory (size_t size, uint32_t permissions, Error& error)
{
    // FIXME: vm_allocate allocates a page at a time, so we should use
    // host_page_size to get the host page size and then parcel out the
    // page we get back until it is filled.
    // FIXME: Add log messages.

    kern_return_t kret;
    mach_vm_address_t addr;
    LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_MEMORY));

    kret = ::mach_vm_allocate (GetTaskPort(), &addr, size, TRUE);
    if (kret == KERN_SUCCESS)
    {
        // Set the protections:
        vm_prot_t mach_prot = 0;
        if (permissions & lldb::ePermissionsReadable)
            mach_prot |= VM_PROT_READ;
        if (permissions & lldb::ePermissionsWritable)
            mach_prot |= VM_PROT_WRITE;
        if (permissions & lldb::ePermissionsExecutable)
            mach_prot |= VM_PROT_EXECUTE;

        kret = ::mach_vm_protect (GetTaskPort(), addr, size, 0, mach_prot);
        if (kret == KERN_SUCCESS)
        {
            if (log)
                log->Printf("Allocated memory at addr = 0x%16.16llx, size = %zu, prot = 0x%x)", (uint64_t) addr, size, mach_prot);
            m_allocations.insert (std::make_pair(addr, size));
            return (lldb::addr_t) addr;
        }
        else
        {
            if (log)
                log->Printf("Failed to set protections on memory at addr = 0x%16.16llx, size = %zu), prot = 0x%x", (uint64_t) addr, size, mach_prot);
            kret = ::mach_vm_deallocate (GetTaskPort(), addr, size);
            return LLDB_INVALID_ADDRESS;
        }
    }
    else
    {
        if (log)
            log->Printf("Failed to set allocate memory: size = %zu)", size);
        return LLDB_INVALID_ADDRESS;
    }
}

//----------------------------------------------------------------------
// MachTask::DeallocateMemory
//----------------------------------------------------------------------
Error
MachTask::DeallocateMemory (lldb::addr_t ptr)
{
    Error error;
    // We have to stash away sizes for the allocations...
    allocation_collection::iterator pos, end = m_allocations.end();
    for (pos = m_allocations.begin(); pos != end; pos++)
    {
        if ((*pos).first == ptr)
        {
            m_allocations.erase (pos);
            error = ::mach_vm_deallocate (GetTaskPort(), (vm_address_t) ptr, (*pos).second);
            return error;
        }
    }
    error.SetErrorStringWithFormat("no memory allocated at 0x%llx", (uint64_t)ptr);
    return error;
}

//----------------------------------------------------------------------
// MachTask::TaskPortForProcessID
//----------------------------------------------------------------------
task_t
MachTask::GetTaskPortForProcessID (Error &err)
{
    err.Clear();
    if (m_task == TASK_NULL && m_process != NULL)
        m_task = MachTask::GetTaskPortForProcessID(m_process->GetID(), err);
    return m_task;
}

//----------------------------------------------------------------------
// MachTask::TaskPortForProcessID
//----------------------------------------------------------------------
task_t
MachTask::GetTaskPortForProcessID (lldb::pid_t pid, Error &err)
{
    task_t task = TASK_NULL;
    if (pid != LLDB_INVALID_PROCESS_ID)
    {
        mach_port_t task_self = mach_task_self ();
        err = ::task_for_pid ( task_self, pid, &task);
        LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_TASK));
        if (log || err.Fail())
        {
            err.PutToLog(log.get(), "::task_for_pid ( target_tport = 0x%4.4x, pid = %d, task => 0x%4.4x ) %u/%u %u/%u", task_self, pid, task, getuid(), geteuid(), getgid(), getegid());
        }
    }
    return task;
}


//----------------------------------------------------------------------
// MachTask::BasicInfo
//----------------------------------------------------------------------
kern_return_t
MachTask::BasicInfo(struct task_basic_info *info) const
{
    return BasicInfo (GetTaskPort(), info);
}

//----------------------------------------------------------------------
// MachTask::BasicInfo
//----------------------------------------------------------------------
kern_return_t
MachTask::BasicInfo(task_t task, struct task_basic_info *info)
{
    if (info == NULL)
        return KERN_INVALID_ARGUMENT;

    Error err;
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    err = ::task_info (task, TASK_BASIC_INFO, (task_info_t)info, &count);
    LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_TASK));
    if (log || err.Fail())
        err.PutToLog(log.get(), "::task_info ( target_task = 0x%4.4x, flavor = TASK_BASIC_INFO, task_info_out => %p, task_info_outCnt => %u )", task, info, count);
    if (log && log->GetMask().Test(PD_LOG_VERBOSE) && err.Success())
    {
        float user = (float)info->user_time.seconds + (float)info->user_time.microseconds / 1000000.0f;
        float system = (float)info->user_time.seconds + (float)info->user_time.microseconds / 1000000.0f;
        log->Printf ("task_basic_info = { suspend_count = %i, virtual_size = 0x%8.8x, resident_size = 0x%8.8x, user_time = %f, system_time = %f }",
                    info->suspend_count, info->virtual_size, info->resident_size, user, system);
    }
    return err.GetError();
}


//----------------------------------------------------------------------
// MachTask::IsValid
//
// Returns true if a task is a valid task port for a current process.
//----------------------------------------------------------------------
bool
MachTask::IsValid () const
{
    return MachTask::IsValid(GetTaskPort());
}

//----------------------------------------------------------------------
// MachTask::IsValid
//
// Returns true if a task is a valid task port for a current process.
//----------------------------------------------------------------------
bool
MachTask::IsValid (task_t task)
{
    if (task != TASK_NULL)
    {
        struct task_basic_info task_info;
        return BasicInfo(task, &task_info) == KERN_SUCCESS;
    }
    return false;
}


bool
MachTask::StartExceptionThread(Error &err)
{
    LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_EXCEPTIONS));

    if (log)
        log->Printf ("MachTask::%s ( )", __FUNCTION__);
    task_t task = GetTaskPortForProcessID(err);
    if (MachTask::IsValid(task))
    {
        // Got the mach port for the current process
        mach_port_t task_self = mach_task_self ();

        // Allocate an exception port that we will use to track our child process
        err = ::mach_port_allocate (task_self, MACH_PORT_RIGHT_RECEIVE, &m_exception_port);
        if (log || err.Fail())
            err.PutToLog(log.get(), "::mach_port_allocate (task_self=0x%4.4x, MACH_PORT_RIGHT_RECEIVE, &m_exception_port => 0x%4.4x)",
                         task_self, m_exception_port);
        if (err.Fail())
            return false;

        // Add the ability to send messages on the new exception port
        err = ::mach_port_insert_right (task_self, m_exception_port, m_exception_port, MACH_MSG_TYPE_MAKE_SEND);
        if (log || err.Fail())
            err.PutToLog(log.get(), "::mach_port_insert_right (task_self=0x%4.4x, m_exception_port=0x%4.4x, m_exception_port=0x%4.4x, MACH_MSG_TYPE_MAKE_SEND)",
                         task_self, m_exception_port, m_exception_port);
        if (err.Fail())
            return false;

        // Save the original state of the exception ports for our child process
        err = SaveExceptionPortInfo();

        // Set the ability to get all exceptions on this port
        err = ::task_set_exception_ports (task, EXC_MASK_ALL, m_exception_port, EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES, THREAD_STATE_NONE);
        if (log || err.Fail())
            err.PutToLog(log.get(), "::task_set_exception_ports (task, EXC_MASK_ALL, m_exception_port, EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES, THREAD_STATE_NONE)");
        if (err.Fail())
            return false;

        // Create the exception thread
        char thread_name[256];
        ::snprintf (thread_name, sizeof(thread_name), "<lldb.process.process-macosx.mach-exception-%d>", m_process->GetID());
        m_exception_thread = Host::ThreadCreate (thread_name, MachTask::ExceptionThread, this, &err);

        return err.Success();
    }
    return false;
}

kern_return_t
MachTask::ShutDownExceptionThread()
{
    Error err;

    if (m_exception_thread == NULL)
        return KERN_SUCCESS;

    err = RestoreExceptionPortInfo();

    LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_EXCEPTIONS));

    // NULL our our exception port and let our exception thread exit
    mach_port_t exception_port = m_exception_port;
    m_exception_port = NULL;

    Host::ThreadCancel (m_exception_thread, &err);
    if (log || err.Fail())
        err.PutToLog(log.get(), "Host::ThreadCancel ( thread = %p )", m_exception_thread);

    Host::ThreadJoin (m_exception_thread, NULL, &err);
    if (log || err.Fail())
        err.PutToLog(log.get(), "Host::ThreadJoin ( thread = %p, result_ptr = NULL)", m_exception_thread);

    // Deallocate our exception port that we used to track our child process
    mach_port_t task_self = mach_task_self ();
    err = ::mach_port_deallocate (task_self, exception_port);
    if (log || err.Fail())
        err.PutToLog(log.get(), "::mach_port_deallocate ( task = 0x%4.4x, name = 0x%4.4x )", task_self, exception_port);
    exception_port = NULL;

    Clear();
    return err.GetError();
}


void *
MachTask::ExceptionThread (void *arg)
{
    if (arg == NULL)
        return NULL;

    MachTask *mach_task = (MachTask*) arg;
    ProcessMacOSX *mach_proc = mach_task->Process();
    LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_EXCEPTIONS));
    if (log)
        log->Printf ("MachTask::%s (arg = %p) thread starting...", __FUNCTION__, arg);

    // We keep a count of the number of consecutive exceptions received so
    // we know to grab all exceptions without a timeout. We do this to get a
    // bunch of related exceptions on our exception port so we can process
    // then together. When we have multiple threads, we can get an exception
    // per thread and they will come in consecutively. The main loop in this
    // thread can stop periodically if needed to service things related to this
    // process.
    // flag set in the options, so we will wait forever for an exception on
    // our exception port. After we get one exception, we then will use the
    // MACH_RCV_TIMEOUT option with a zero timeout to grab all other current
    // exceptions for our process. After we have received the last pending
    // exception, we will get a timeout which enables us to then notify
    // our main thread that we have an exception bundle available. We then wait
    // for the main thread to tell this exception thread to start trying to get
    // exceptions messages again and we start again with a mach_msg read with
    // infinite timeout.
    uint32_t num_exceptions_received = 0;
    Error err;
    task_t task = mach_task->GetTaskPort();
    mach_msg_timeout_t periodic_timeout = 1000;

#if defined (__arm__)
    mach_msg_timeout_t watchdog_elapsed = 0;
    mach_msg_timeout_t watchdog_timeout = 60 * 1000;
    lldb::pid_t pid = mach_proc->GetID();
    CFReleaser<SBSWatchdogAssertionRef> watchdog;

    if (mach_proc->ProcessUsingSpringBoard())
    {
        // Request a renewal for every 60 seconds if we attached using SpringBoard
        watchdog.reset(::SBSWatchdogAssertionCreateForPID(NULL, pid, 60));
        if (log)
            log->Printf ("::SBSWatchdogAssertionCreateForPID (NULL, %4.4x, 60 ) => %p", pid, watchdog.get());

        if (watchdog.get())
        {
            ::SBSWatchdogAssertionRenew (watchdog.get());

            CFTimeInterval watchdogRenewalInterval = ::SBSWatchdogAssertionGetRenewalInterval (watchdog.get());
            if (log)
                log->Printf ("::SBSWatchdogAssertionGetRenewalInterval ( %p ) => %g seconds", watchdog.get(), watchdogRenewalInterval);
            if (watchdogRenewalInterval > 0.0)
            {
                watchdog_timeout = (mach_msg_timeout_t)watchdogRenewalInterval * 1000;
                if (watchdog_timeout > 3000)
                    watchdog_timeout -= 1000;   // Give us a second to renew our timeout
                else if (watchdog_timeout > 1000)
                    watchdog_timeout -= 250;    // Give us a quarter of a second to renew our timeout
            }
        }
        if (periodic_timeout == 0 || periodic_timeout > watchdog_timeout)
            periodic_timeout = watchdog_timeout;
    }
#endif  // #if defined (__arm__)

    while (mach_task->ExceptionPortIsValid())
    {
        //::pthread_testcancel ();

        MachException::Message exception_message;


        if (num_exceptions_received > 0)
        {
            // No timeout, just receive as many exceptions as we can since we already have one and we want
            // to get all currently available exceptions for this task
            err = exception_message.Receive(mach_task->ExceptionPort(), MACH_RCV_MSG | MACH_RCV_INTERRUPT | MACH_RCV_TIMEOUT, 0);
        }
        else if (periodic_timeout > 0)
        {
            // We need to stop periodically in this loop, so try and get a mach message with a valid timeout (ms)
            err = exception_message.Receive(mach_task->ExceptionPort(), MACH_RCV_MSG | MACH_RCV_INTERRUPT | MACH_RCV_TIMEOUT, periodic_timeout);
        }
        else
        {
            // We don't need to parse all current exceptions or stop periodically,
            // just wait for an exception forever.
            err = exception_message.Receive(mach_task->ExceptionPort(), MACH_RCV_MSG | MACH_RCV_INTERRUPT, 0);
        }

        if (err.GetError() == MACH_RCV_INTERRUPTED)
        {
            log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_EXCEPTIONS);
            // If we have no task port we should exit this thread
            if (!mach_task->ExceptionPortIsValid())
            {
                if (log)
                    log->Printf ("thread cancelled...");
                break;
            }

            // Make sure our task is still valid
            if (MachTask::IsValid(task))
            {
                // Task is still ok
                if (log)
                    log->Printf ("interrupted, but task still valid, continuing...");
                continue;
            }
            else
            {
                if (log)
                    log->Printf ("task has exited...");
                mach_proc->SetPrivateState (eStateExited);
                // Our task has died, exit the thread.
                break;
            }
        }
        else if (err.GetError() == MACH_RCV_TIMED_OUT)
        {
            if (num_exceptions_received > 0)
            {
                log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_EXCEPTIONS);
            
                // We were receiving all current exceptions with a timeout of zero
                // it is time to go back to our normal looping mode
                num_exceptions_received = 0;

                // Notify our main thread we have a complete exception message
                // bundle available.
                mach_proc->ExceptionMessageBundleComplete();

                // in case we use a timeout value when getting exceptions...
                // Make sure our task is still valid
                if (MachTask::IsValid(task))
                {
                    // Task is still ok
                    if (log)
                        log->Printf ("got a timeout, continuing...");
                    continue;
                }
                else
                {
                    if (log)
                        log->Printf ("task has exited...");
                    mach_proc->SetPrivateState (eStateExited);
                    // Our task has died, exit the thread.
                    break;
                }
                continue;
            }

#if defined (__arm__)
            if (watchdog.get())
            {
                watchdog_elapsed += periodic_timeout;
                if (watchdog_elapsed >= watchdog_timeout)
                {
                    LogIf(PD_LOG_TASK, "SBSWatchdogAssertionRenew ( %p )", watchdog.get());
                    ::SBSWatchdogAssertionRenew (watchdog.get());
                    watchdog_elapsed = 0;
                }
            }
#endif
        }
        else if (err.GetError() != KERN_SUCCESS)
        {
            log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_EXCEPTIONS);
            if (log)
                log->Printf ("got some other error, do something about it??? nah, continuing for now...");
            // TODO: notify of error?
        }
        else
        {
            if (exception_message.CatchExceptionRaise())
            {
                ++num_exceptions_received;
                mach_proc->ExceptionMessageReceived(exception_message);
            }
        }
    }

#if defined (__arm__)
    if (watchdog.get())
    {
        // TODO: change SBSWatchdogAssertionRelease to SBSWatchdogAssertionCancel when we
        // all are up and running on systems that support it. The SBS framework has a #define
        // that will forward SBSWatchdogAssertionRelease to SBSWatchdogAssertionCancel for now
        // so it should still build either way.
        LogIf(PD_LOG_TASK, "::SBSWatchdogAssertionRelease(%p)", watchdog.get());
        ::SBSWatchdogAssertionRelease (watchdog.get());
    }
#endif  // #if defined (__arm__)

    log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_EXCEPTIONS);
    if (log)
        log->Printf ("MachTask::%s (arg = %p) thread exiting...", __FUNCTION__, arg);
    return NULL;
}

lldb::addr_t
MachTask::GetDYLDAllImageInfosAddress ()
{
#ifdef TASK_DYLD_INFO
    task_dyld_info_data_t dyld_info;
    mach_msg_type_number_t count = TASK_DYLD_INFO_COUNT;
    Error err;
        // The actual task shouldn't matter for the DYLD info, so lets just use ours
    kern_return_t kret = ::task_info (GetTaskPortForProcessID (err), TASK_DYLD_INFO, (task_info_t)&dyld_info, &count);
    if (kret == KERN_SUCCESS)
    {
        // We now have the address of the all image infos structure
        return dyld_info.all_image_info_addr;
    }
#endif
    return LLDB_INVALID_ADDRESS;
}





