//===-- TargetList.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/State.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/TargetList.h"

using namespace lldb;
using namespace lldb_private;


//----------------------------------------------------------------------
// TargetList constructor
//----------------------------------------------------------------------
TargetList::TargetList() :
    Broadcaster("TargetList"),
    m_target_list(),
    m_target_list_mutex (Mutex::eMutexTypeRecursive),
    m_selected_target_idx (0)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
TargetList::~TargetList()
{
    Mutex::Locker locker(m_target_list_mutex);
    m_target_list.clear();
}

Error
TargetList::CreateTarget
(
    Debugger &debugger,
    const FileSpec& file,
    const ArchSpec& arch,
    bool get_dependent_files,
    TargetSP &target_sp
)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "TargetList::CreateTarget (file = '%s/%s', arch = '%s')",
                        file.GetDirectory().AsCString(),
                        file.GetFilename().AsCString(),
                        arch.GetArchitectureName());
    Error error;

    PlatformSP platform_sp (debugger.GetPlatformList().GetSelectedPlatform ());
    
    if (file)
    {
        ModuleSP exe_module_sp;
        FileSpec resolved_file(file);
        
        if (platform_sp)
            error = platform_sp->ResolveExecutable (file, arch, exe_module_sp);

        if (error.Success() && exe_module_sp)
        {
            if (exe_module_sp->GetObjectFile() == NULL)
            {
                if (arch.IsValid())
                {
                    error.SetErrorStringWithFormat("\"%s%s%s\" doesn't contain architecture %s",
                                                   file.GetDirectory().AsCString(),
                                                   file.GetDirectory() ? "/" : "",
                                                   file.GetFilename().AsCString(),
                                                   arch.GetArchitectureName());
                }
                else
                {
                    error.SetErrorStringWithFormat("unsupported file type \"%s%s%s\"",
                                                   file.GetDirectory().AsCString(),
                                                   file.GetDirectory() ? "/" : "",
                                                   file.GetFilename().AsCString());
                }
                return error;
            }
            target_sp.reset(new Target(debugger, arch, platform_sp));
            target_sp->SetExecutableModule (exe_module_sp, get_dependent_files);
        }
    }
    else
    {
        // No file was specified, just create an empty target with any arch
        // if a valid arch was specified
        target_sp.reset(new Target(debugger, arch, platform_sp));
    }

    if (target_sp)
    {
        target_sp->UpdateInstanceName();        

        Mutex::Locker locker(m_target_list_mutex);
        m_selected_target_idx = m_target_list.size();
        m_target_list.push_back(target_sp);
    }

    return error;
}

bool
TargetList::DeleteTarget (TargetSP &target_sp)
{
    Mutex::Locker locker(m_target_list_mutex);
    collection::iterator pos, end = m_target_list.end();

    for (pos = m_target_list.begin(); pos != end; ++pos)
    {
        if (pos->get() == target_sp.get())
        {
            m_target_list.erase(pos);
            return true;
        }
    }
    return false;
}


TargetSP
TargetList::FindTargetWithExecutableAndArchitecture
(
    const FileSpec &exe_file_spec,
    const ArchSpec *exe_arch_ptr
) const
{
    Mutex::Locker locker (m_target_list_mutex);
    TargetSP target_sp;
    bool full_match = exe_file_spec.GetDirectory();

    collection::const_iterator pos, end = m_target_list.end();
    for (pos = m_target_list.begin(); pos != end; ++pos)
    {
        Module *exe_module = (*pos)->GetExecutableModulePointer();

        if (exe_module)
        {
            if (FileSpec::Equal (exe_file_spec, exe_module->GetFileSpec(), full_match))
            {
                if (exe_arch_ptr)
                {
                    if (*exe_arch_ptr != exe_module->GetArchitecture())
                        continue;
                }
                target_sp = *pos;
                break;
            }
        }
    }
    return target_sp;
}

TargetSP
TargetList::FindTargetWithProcessID (lldb::pid_t pid) const
{
    Mutex::Locker locker(m_target_list_mutex);
    TargetSP target_sp;
    collection::const_iterator pos, end = m_target_list.end();
    for (pos = m_target_list.begin(); pos != end; ++pos)
    {
        Process* process = (*pos)->GetProcessSP().get();
        if (process && process->GetID() == pid)
        {
            target_sp = *pos;
            break;
        }
    }
    return target_sp;
}


TargetSP
TargetList::FindTargetWithProcess (Process *process) const
{
    TargetSP target_sp;
    if (process)
    {
        Mutex::Locker locker(m_target_list_mutex);
        collection::const_iterator pos, end = m_target_list.end();
        for (pos = m_target_list.begin(); pos != end; ++pos)
        {
            if (process == (*pos)->GetProcessSP().get())
            {
                target_sp = *pos;
                break;
            }
        }
    }
    return target_sp;
}

TargetSP
TargetList::GetTargetSP (Target *target) const
{
    TargetSP target_sp;
    if (target)
    {
        Mutex::Locker locker(m_target_list_mutex);
        collection::const_iterator pos, end = m_target_list.end();
        for (pos = m_target_list.begin(); pos != end; ++pos)
        {
            if (target == (*pos).get())
            {
                target_sp = *pos;
                break;
            }
        }
    }
    return target_sp;
}

uint32_t
TargetList::SendAsyncInterrupt (lldb::pid_t pid)
{
    uint32_t num_async_interrupts_sent = 0;

    if (pid != LLDB_INVALID_PROCESS_ID)
    {
        TargetSP target_sp(FindTargetWithProcessID (pid));
        if (target_sp.get())
        {
            Process* process = target_sp->GetProcessSP().get();
            if (process)
            {
                process->BroadcastEvent (Process::eBroadcastBitInterrupt, NULL);
                ++num_async_interrupts_sent;
            }
        }
    }
    else
    {
        // We don't have a valid pid to broadcast to, so broadcast to the target
        // list's async broadcaster...
        BroadcastEvent (Process::eBroadcastBitInterrupt, NULL);
    }

    return num_async_interrupts_sent;
}

uint32_t
TargetList::SignalIfRunning (lldb::pid_t pid, int signo)
{
    uint32_t num_signals_sent = 0;
    Process *process = NULL;
    if (pid == LLDB_INVALID_PROCESS_ID)
    {
        // Signal all processes with signal
        Mutex::Locker locker(m_target_list_mutex);
        collection::iterator pos, end = m_target_list.end();
        for (pos = m_target_list.begin(); pos != end; ++pos)
        {
            process = (*pos)->GetProcessSP().get();
            if (process)
            {
                if (process->IsAlive())
                {
                    ++num_signals_sent;
                    process->Signal (signo);
                }
            }
        }
    }
    else
    {
        // Signal a specific process with signal
        TargetSP target_sp(FindTargetWithProcessID (pid));
        if (target_sp.get())
        {
            process = target_sp->GetProcessSP().get();
            if (process)
            {
                if (process->IsAlive())
                {
                    ++num_signals_sent;
                    process->Signal (signo);
                }
            }
        }
    }
    return num_signals_sent;
}

int
TargetList::GetNumTargets () const
{
    Mutex::Locker locker (m_target_list_mutex);
    return m_target_list.size();
}

lldb::TargetSP
TargetList::GetTargetAtIndex (uint32_t idx) const
{
    TargetSP target_sp;
    Mutex::Locker locker (m_target_list_mutex);
    if (idx < m_target_list.size())
        target_sp = m_target_list[idx];
    return target_sp;
}

uint32_t
TargetList::SetSelectedTarget (Target* target)
{
    Mutex::Locker locker (m_target_list_mutex);
    collection::const_iterator pos,
        begin = m_target_list.begin(),
        end = m_target_list.end();
    for (pos = begin; pos != end; ++pos)
    {
        if (pos->get() == target)
        {
            m_selected_target_idx = std::distance (begin, pos);
            return m_selected_target_idx;
        }
    }
    m_selected_target_idx = 0;
    return m_selected_target_idx;
}

lldb::TargetSP
TargetList::GetSelectedTarget ()
{
    Mutex::Locker locker (m_target_list_mutex);
    if (m_selected_target_idx >= m_target_list.size())
        m_selected_target_idx = 0;
    return GetTargetAtIndex (m_selected_target_idx);
}
