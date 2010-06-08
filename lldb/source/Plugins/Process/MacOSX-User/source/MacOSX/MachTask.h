//===-- MachTask.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __MachTask_h__
#define __MachTask_h__

// C Includes
// C++ Includes
#include <map>
// Other libraries and framework includes
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/socket.h>

// Project includes
#include "MachException.h"
#include "MachVMMemory.h"

class ProcessMacOSX;

class MachTask
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    MachTask (ProcessMacOSX *process);

    virtual
    ~MachTask ();

    void
    Clear ();

    kern_return_t
    Suspend ();

    kern_return_t
    Resume ();

    int32_t
    GetSuspendCount () const;

    size_t
    ReadMemory (lldb::addr_t addr, void *buf, size_t size, lldb_private::Error& error);

    size_t
    WriteMemory (lldb::addr_t addr, const void *buf, size_t size, lldb_private::Error& error);

    lldb::addr_t
    AllocateMemory (size_t size, uint32_t permissions, lldb_private::Error& error);

    lldb_private::Error
    DeallocateMemory (lldb::addr_t addr);

    mach_port_t
    ExceptionPort () const;

    bool
    ExceptionPortIsValid () const;

    kern_return_t
    SaveExceptionPortInfo ();

    kern_return_t
    RestoreExceptionPortInfo ();

    kern_return_t
    ShutDownExceptionThread ();

    bool
    StartExceptionThread (lldb_private::Error &err);

    lldb::addr_t
    GetDYLDAllImageInfosAddress ();

    kern_return_t
    BasicInfo (struct task_basic_info *info) const;

    static kern_return_t
    BasicInfo (task_t task, struct task_basic_info *info);

    bool
    IsValid () const;

    static bool
    IsValid (task_t task);

    static void *
    ExceptionThread (void *arg);

    task_t
    GetTaskPort () const
    {
        return m_task;
    }

    task_t
    GetTaskPortForProcessID (lldb_private::Error &err);

    static task_t
    GetTaskPortForProcessID (lldb::pid_t pid, lldb_private::Error &err);

    ProcessMacOSX *
    Process ()
    {
        return m_process;
    }

    const ProcessMacOSX *
    Process () const
    {
        return m_process;
    }

protected:
    ProcessMacOSX * m_process;                  // The mach process that owns this MachTask
    task_t m_task;
    MachVMMemory m_vm_memory;                   // Special mach memory reading class that will take care of watching for page and region boundaries
    MachException::PortInfo m_exc_port_info;    // Saved settings for all exception ports
    lldb::thread_t m_exception_thread;          // Thread ID for the exception thread in case we need it
    mach_port_t m_exception_port;               // Exception port on which we will receive child exceptions

    // Maybe sort this by address and use find?
    typedef std::map<vm_address_t,size_t> allocation_collection;
    allocation_collection m_allocations;

private:
    DISALLOW_COPY_AND_ASSIGN (MachTask);
};

#endif  // __MachTask_h__
