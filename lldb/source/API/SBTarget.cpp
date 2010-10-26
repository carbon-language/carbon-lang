//===-- SBTarget.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBTarget.h"

#include "lldb/lldb-include.h"

#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBStream.h"
#include "lldb/Breakpoint/BreakpointID.h"
#include "lldb/Breakpoint/BreakpointIDList.h"
#include "lldb/Breakpoint/BreakpointList.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/AddressResolver.h"
#include "lldb/Core/AddressResolverName.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Core/STLUtils.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetList.h"

#include "lldb/Interpreter/CommandReturnObject.h"
#include "../source/Commands/CommandObjectBreakpoint.h"

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBBreakpoint.h"

using namespace lldb;
using namespace lldb_private;

#define DEFAULT_DISASM_BYTE_SIZE 32

//----------------------------------------------------------------------
// SBTarget constructor
//----------------------------------------------------------------------
SBTarget::SBTarget ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBTarget::SBTarget () ==> this = %p", this);
}

SBTarget::SBTarget (const SBTarget& rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);
    
    if (log)
        log->Printf ("SBTarget::SBTarget (const SBTarget &rhs) rhs.m_opaque_sp.get() = %p ==> this = %p",
                     rhs.m_opaque_sp.get(), this);
}

SBTarget::SBTarget(const TargetSP& target_sp) :
    m_opaque_sp (target_sp)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBTarget::SBTarget (const TargetSP &target_sp)  target_sp.get() = %p ==> this = %p",
                     target_sp.get(), this);
}

const SBTarget&
SBTarget::Assign (const SBTarget& rhs)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::Assign (const SBTarget &rhs)  rhs = %p", &rhs);

    if (this != &rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
    }

    if (log)
        log->Printf ("SBTarget::Assign ==> SBTarget (this = %p, m_opaque_sp.get() = %p)", this, m_opaque_sp.get());

    return *this;
}


//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SBTarget::~SBTarget()
{
}

bool
SBTarget::IsValid () const
{
    return m_opaque_sp.get() != NULL;
}

SBProcess
SBTarget::GetProcess ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::GetProcess ()");

    SBProcess sb_process;
    if (m_opaque_sp)
        sb_process.SetProcess (m_opaque_sp->GetProcessSP());

    if (log)
    {
        SBStream sstr;
        sb_process.GetDescription (sstr);
        log->Printf ("SBTarget::GetProcess ==> SBProcess (this = %p, '%s')", &sb_process, sstr.GetData());
    }

    return sb_process;
}

SBDebugger
SBTarget::GetDebugger () const
{
    SBDebugger debugger;
    if (m_opaque_sp)
        debugger.reset (m_opaque_sp->GetDebugger().GetSP());
    return debugger;
}


// DEPRECATED
SBProcess
SBTarget::CreateProcess ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::CreateProcess ()");

    SBProcess sb_process;

    if (m_opaque_sp)
        sb_process.SetProcess (m_opaque_sp->CreateProcess (m_opaque_sp->GetDebugger().GetListener()));

    if (log)
    {
        SBStream sstr;
        sb_process.GetDescription (sstr);
        log->Printf ("SBTarget::CreateProcess ==> SBProcess (this = %p, '%s')", &sb_process, sstr.GetData());
    }

    return sb_process;
}


SBProcess
SBTarget::LaunchProcess
(
    char const **argv,
    char const **envp,
    const char *tty,
    uint32_t launch_flags,
    bool stop_at_entry
)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
    {
        log->Printf ("SBTarget::LaunchProcess (char const **argv, char const **envp, const char *tty, "
                     "uint32_t launch_flags, bool stop_at_entry)");

        if (!argv)
            log->Printf ("argv:  NULL");
        else
        {
            for (int i = 0; argv[i]; ++i)
                log->Printf ("     %s", argv[i]);
        }

        if (!envp)
            log->Printf ("envp: NULL");
        else
        {
            for (int i = 0; envp[i]; ++i)
                log->Printf ("     %s", envp[i]);
        }

        log->Printf ("     tty = %s, launch_flags = %d, stop_at_entry = %s", tty, launch_flags, (stop_at_entry ? 
                                                                                                 "true" :
                                                                                                 "false"));    
    }

    SBError sb_error;    
    SBProcess sb_process = Launch (argv, envp, tty, launch_flags, stop_at_entry, sb_error);

    if (log)
    {
        SBStream sstr;
        sb_process.GetDescription (sstr);
        log->Printf ("SBTarget::LaunchProcess ==> SBProcess (this = %p, '%s')", this, sstr.GetData());
    }

    return sb_process;
}

SBProcess
SBTarget::Launch
(
    char const **argv,
    char const **envp,
    const char *tty,
    uint32_t launch_flags,
    bool stop_at_entry,
    SBError &error
)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
    {
        log->Printf ("SBTarget::Launch (char const **argv, char const **envp, const char *tty, uint32_t launch_flag,"
                     "bool stop_at_entry, SBError error)");
        if (!argv)
            log->Printf ("argv:  NULL");
        else
        {
            for (int i = 0; argv[i]; ++i)
                log->Printf ("     %s", argv[i]);
        }

        if (!envp)
            log->Printf ("envp: NULL");
        else
        {
            for (int i = 0; envp[i]; ++i)
                log->Printf ("     %s", envp[i]);
        }

        log->Printf ("     tty = %s, launch_flags = %d, stop_at_entry = %s, error (this = %p)", tty, launch_flags, 
                     (stop_at_entry ? "true" : "false"), &error);    
    }

    SBProcess sb_process;
    if (m_opaque_sp)
    {
        // DEPRECATED, this will change when CreateProcess is removed...
        if (m_opaque_sp->GetProcessSP())
        {
            sb_process.SetProcess(m_opaque_sp->GetProcessSP());
        }
        else
        {
            // When launching, we always want to create a new process When
            // SBTarget::CreateProcess is removed, this will always happen.
            sb_process.SetProcess (m_opaque_sp->CreateProcess (m_opaque_sp->GetDebugger().GetListener()));
        }

        if (sb_process.IsValid())
        {
            error.SetError (sb_process->Launch (argv, envp, launch_flags, tty, tty, tty));
            if (error.Success())
            {
                // We we are stopping at the entry point, we can return now!
                if (stop_at_entry)
                    return sb_process;
                
                // Make sure we are stopped at the entry
                StateType state = sb_process->WaitForProcessToStop (NULL);
                if (state == eStateStopped)
                {
                    // resume the process to skip the entry point
                    error.SetError (sb_process->Resume());
                    if (error.Success())
                    {
                        // If we are doing synchronous mode, then wait for the
                        // process to stop yet again!
                        if (m_opaque_sp->GetDebugger().GetAsyncExecution () == false)
                            sb_process->WaitForProcessToStop (NULL);
                    }
                }
            }
        }
        else
        {
            error.SetErrorString ("unable to create lldb_private::Process");
        }
    }
    else
    {
        error.SetErrorString ("SBTarget is invalid");
    }

    if (log)
    {
        SBStream sstr;
        sb_process.GetDescription (sstr);
        log->Printf ("SBTarget::Launch ==> SBProceess (this = %p, '%s')", &sb_process, sstr.GetData());
    }

    return sb_process;
}


lldb::SBProcess
SBTarget::AttachToProcessWithID 
(
    lldb::pid_t pid,// The process ID to attach to
    SBError& error  // An error explaining what went wrong if attach fails
)
{
    SBProcess sb_process;
    if (m_opaque_sp)
    {
        // DEPRECATED, this will change when CreateProcess is removed...
        if (m_opaque_sp->GetProcessSP())
        {
            sb_process.SetProcess(m_opaque_sp->GetProcessSP());
        }
        else
        {
            // When launching, we always want to create a new process When
            // SBTarget::CreateProcess is removed, this will always happen.
            sb_process.SetProcess (m_opaque_sp->CreateProcess (m_opaque_sp->GetDebugger().GetListener()));
        }

        if (sb_process.IsValid())
        {
            error.SetError (sb_process->Attach (pid));
        }
        else
        {
            error.SetErrorString ("unable to create lldb_private::Process");
        }
    }
    else
    {
        error.SetErrorString ("SBTarget is invalid");
    }
    return sb_process;

}

lldb::SBProcess
SBTarget::AttachToProcessWithName 
(
    const char *name,   // basename of process to attach to
    bool wait_for,      // if true wait for a new instance of "name" to be launched
    SBError& error      // An error explaining what went wrong if attach fails
)
{
    SBProcess sb_process;
    if (m_opaque_sp)
    {
        // DEPRECATED, this will change when CreateProcess is removed...
        if (m_opaque_sp->GetProcessSP())
        {
            sb_process.SetProcess(m_opaque_sp->GetProcessSP());
        }
        else
        {
            // When launching, we always want to create a new process When
            // SBTarget::CreateProcess is removed, this will always happen.
            sb_process.SetProcess (m_opaque_sp->CreateProcess (m_opaque_sp->GetDebugger().GetListener()));
        }

        if (sb_process.IsValid())
        {
            error.SetError (sb_process->Attach (name, wait_for));
        }
        else
        {
            error.SetErrorString ("unable to create lldb_private::Process");
        }
    }
    else
    {
        error.SetErrorString ("SBTarget is invalid");
    }
    return sb_process;

}

SBFileSpec
SBTarget::GetExecutable ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::GetExecutable ()");

    SBFileSpec exe_file_spec;
    if (m_opaque_sp)
    {
        ModuleSP exe_module_sp (m_opaque_sp->GetExecutableModule ());
        if (exe_module_sp)
            exe_file_spec.SetFileSpec (exe_module_sp->GetFileSpec());
    }

    if (log)
    {
        if (exe_file_spec.Exists())
        {
            SBStream sstr;
            exe_file_spec.GetDescription (sstr);
            log->Printf ("SBTarget::GetExecutable ==> SBFileSpec (this = %p, '%s')", &exe_file_spec, sstr.GetData());
        }
        else
            log->Printf ("SBTarget::GetExecutable ==> SBFileSpec (this = %p, 'Unable to find valid file')",
                         &exe_file_spec);
    }

    return exe_file_spec;
}


bool
SBTarget::DeleteTargetFromList (TargetList *list)
{
    if (m_opaque_sp)
        return list->DeleteTarget (m_opaque_sp);
    else
        return false;
}

bool
SBTarget::operator == (const SBTarget &rhs) const
{
    return m_opaque_sp.get() == rhs.m_opaque_sp.get();
}

bool
SBTarget::operator != (const SBTarget &rhs) const
{
    return m_opaque_sp.get() != rhs.m_opaque_sp.get();
}

lldb_private::Target *
SBTarget::operator ->() const
{
    return m_opaque_sp.get();
}

lldb_private::Target *
SBTarget::get() const
{
    return m_opaque_sp.get();
}

void
SBTarget::reset (const lldb::TargetSP& target_sp)
{
    m_opaque_sp = target_sp;
}

SBBreakpoint
SBTarget::BreakpointCreateByLocation (const char *file, uint32_t line)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::BreakpointCreateByLocation (const char *file, uint32_t line) file = '%s', line = %d", 
                     file, line);

    SBBreakpoint sb_bp;
    if (file != NULL && line != 0)
        sb_bp = BreakpointCreateByLocation (SBFileSpec (file), line);

    if (log)
    {
        SBStream sstr;
        sb_bp.GetDescription (sstr);
        log->Printf("SBTarget::BreakpointCreateByLocation ==> SBBreakpoint (this = %p, '%s')", &sb_bp, sstr.GetData());
    }

    return sb_bp;
}

SBBreakpoint
SBTarget::BreakpointCreateByLocation (const SBFileSpec &sb_file_spec, uint32_t line)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::BreakpointCreateByLocation (const SBFileSpec &sb_file_spec, uint32_t line) "
                     "sb_file_spec (%p), line = %d)", &sb_file_spec, line);

    SBBreakpoint sb_bp;
    if (m_opaque_sp.get() && line != 0)
        *sb_bp = m_opaque_sp->CreateBreakpoint (NULL, *sb_file_spec, line, true, false);

    if (log)
    {
        SBStream sstr;
        sb_bp.GetDescription (sstr);
        log->Printf ("SBTarget::BreakpointCreateByLocation ==> SBBreakpoint (this = %p, '%s')", &sb_bp, 
                     sstr.GetData());
    }

    return sb_bp;
}

SBBreakpoint
SBTarget::BreakpointCreateByName (const char *symbol_name, const char *module_name)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::BreakpointCreateByName (const char *symbol_name, const char *module_name) "
                     "symbol_name = %s, module_name = %s)", symbol_name, module_name);

    SBBreakpoint sb_bp;
    if (m_opaque_sp.get() && symbol_name && symbol_name[0])
    {
        if (module_name && module_name[0])
        {
            FileSpec module_file_spec(module_name, false);
            *sb_bp = m_opaque_sp->CreateBreakpoint (&module_file_spec, symbol_name, eFunctionNameTypeFull | eFunctionNameTypeBase, false);
        }
        else
        {
            *sb_bp = m_opaque_sp->CreateBreakpoint (NULL, symbol_name, eFunctionNameTypeFull | eFunctionNameTypeBase, false);
        }
    }
    
    if (log)
    {
        SBStream sstr;
        sb_bp.GetDescription (sstr);
        log->Printf ("SBTarget::BreakpointCreateByName ==> SBBreakpoint (this = %p, '%s')", &sb_bp, sstr.GetData());
    }

    return sb_bp;
}

SBBreakpoint
SBTarget::BreakpointCreateByRegex (const char *symbol_name_regex, const char *module_name)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::BreakpointCreateByRegex (const char *symbol_name_regex, const char *module_name) "
                     "symbol_name_regex = %s, module_name = %s)", symbol_name_regex, module_name);

    SBBreakpoint sb_bp;
    if (m_opaque_sp.get() && symbol_name_regex && symbol_name_regex[0])
    {
        RegularExpression regexp(symbol_name_regex);
        
        if (module_name && module_name[0])
        {
            FileSpec module_file_spec(module_name, false);
            
            *sb_bp = m_opaque_sp->CreateBreakpoint (&module_file_spec, regexp, false);
        }
        else
        {
            *sb_bp = m_opaque_sp->CreateBreakpoint (NULL, regexp, false);
        }
    }

    if (log)
    {
        SBStream sstr;
        sb_bp.GetDescription (sstr);
        log->Printf ("SBTarget::BreakpointCreateByRegex ==> SBBreakpoint (this = %p, '%s')", &sb_bp, sstr.GetData());
    }

    return sb_bp;
}



SBBreakpoint
SBTarget::BreakpointCreateByAddress (addr_t address)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::BreakpointCreateByAddress (addr_t address) address = %p", address);

    SBBreakpoint sb_bp;
    if (m_opaque_sp.get())
        *sb_bp = m_opaque_sp->CreateBreakpoint (address, false);
    
    if (log)
    {
        SBStream sstr;
        sb_bp.GetDescription (sstr);
        log->Printf ("SBTarget::BreakpointCreateByAddress ==> SBBreakpoint (this = %p, '%s')", &sb_bp, sstr.GetData());
    }

    return sb_bp;
}

SBBreakpoint
SBTarget::FindBreakpointByID (break_id_t bp_id)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::FindBreakpointByID (break_id_t bp_id) bp_id = %d", bp_id);

    SBBreakpoint sb_breakpoint;
    if (m_opaque_sp && bp_id != LLDB_INVALID_BREAK_ID)
        *sb_breakpoint = m_opaque_sp->GetBreakpointByID (bp_id);

    if (log)
    {
        SBStream sstr;
        sb_breakpoint.GetDescription (sstr);
        log->Printf ("SBTarget::FindBreakpointByID ==> SBBreakpoint (this = %p, '%s'", &bp_id, sstr.GetData());
    }

    return sb_breakpoint;
}

uint32_t
SBTarget::GetNumBreakpoints () const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetBreakpointList().GetSize();
    return 0;
}

SBBreakpoint
SBTarget::GetBreakpointAtIndex (uint32_t idx) const
{
    SBBreakpoint sb_breakpoint;
    if (m_opaque_sp)
        *sb_breakpoint = m_opaque_sp->GetBreakpointList().GetBreakpointAtIndex(idx);
    return sb_breakpoint;
}

bool
SBTarget::BreakpointDelete (break_id_t bp_id)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::BreakpointDelete (break_id_t bp_id) bp_id = %d", bp_id);

    bool result = false;
    if (m_opaque_sp)
        result = m_opaque_sp->RemoveBreakpointByID (bp_id);

    if (log)
    {
        if (result)
            log->Printf ("SBTarget::BreakpointDelete ==> true");
        else
            log->Printf ("SBTarget::BreakpointDelete ==> false");
    }

    return result;
}

bool
SBTarget::EnableAllBreakpoints ()
{
    if (m_opaque_sp)
    {
        m_opaque_sp->EnableAllBreakpoints ();
        return true;
    }
    return false;
}

bool
SBTarget::DisableAllBreakpoints ()
{
    if (m_opaque_sp)
    {
        m_opaque_sp->DisableAllBreakpoints ();
        return true;
    }
    return false;
}

bool
SBTarget::DeleteAllBreakpoints ()
{
    if (m_opaque_sp)
    {
        m_opaque_sp->RemoveAllBreakpoints ();
        return true;
    }
    return false;
}


uint32_t
SBTarget::GetNumModules () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::GetNumModules ()");

    uint32_t num = 0;
    if (m_opaque_sp)
        num =  m_opaque_sp->GetImages().GetSize();

    if (log)
        log->Printf ("SBTarget::GetNumModules ==> %d", num);

    return num;
}

void
SBTarget::Clear ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::Clear ()");

    m_opaque_sp.reset();
}


SBModule
SBTarget::FindModule (const SBFileSpec &sb_file_spec)
{
    SBModule sb_module;
    if (m_opaque_sp && sb_file_spec.IsValid())
        sb_module.SetModule (m_opaque_sp->GetImages().FindFirstModuleForFileSpec (*sb_file_spec, NULL));
    return sb_module;
}

SBModule
SBTarget::GetModuleAtIndex (uint32_t idx)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::GetModuleAtIndex (uint32_t idx) idx = %d", idx);

    SBModule sb_module;
    if (m_opaque_sp)
        sb_module.SetModule(m_opaque_sp->GetImages().GetModuleAtIndex(idx));

    if (log)
    {
        SBStream sstr;
        sb_module.GetDescription (sstr);
        log->Printf ("SBTarget::GetModuleAtIndex ==> SBModule: this = %p, %s", &sb_module, sstr.GetData());
    }

    return sb_module;
}


SBBroadcaster
SBTarget::GetBroadcaster () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBTarget::GetBroadcaster ()");

    SBBroadcaster broadcaster(m_opaque_sp.get(), false);
    
    if (log)
        log->Printf ("SBTarget::GetBroadcaster ==> SBBroadcaster (this = %p)", &broadcaster);

    return broadcaster;
}

void
SBTarget::Disassemble (lldb::addr_t start_addr, lldb::addr_t end_addr, const char *module_name)
{
    if (start_addr == LLDB_INVALID_ADDRESS)
        return;

    FILE *out = m_opaque_sp->GetDebugger().GetOutputFileHandle();
    if (out == NULL)
        return;

    if (m_opaque_sp)
    {
        ModuleSP module_sp;
        if (module_name != NULL)
        {
            FileSpec module_file_spec (module_name, false);
            module_sp = m_opaque_sp->GetImages().FindFirstModuleForFileSpec (module_file_spec, NULL);
        }
        
        AddressRange range;

        // Make sure the process object is alive if we have one (it might be
        // created but we might not be launched yet).
        
        Process *sb_process = m_opaque_sp->GetProcessSP().get();
        if (sb_process && !sb_process->IsAlive())
            sb_process = NULL;
        
        // If we are given a module, then "start_addr" is a file address in
        // that module.
        if (module_sp)
        {
            if (!module_sp->ResolveFileAddress (start_addr, range.GetBaseAddress()))
                range.GetBaseAddress().SetOffset(start_addr);
        }
        else if (m_opaque_sp->GetSectionLoadList().IsEmpty() == false)
        {
            // We don't have a module, se we need to figure out if "start_addr"
            // resolves to anything in a running process.
            if (!m_opaque_sp->GetSectionLoadList().ResolveLoadAddress (start_addr, range.GetBaseAddress()))
                range.GetBaseAddress().SetOffset(start_addr);
        }
        else
        {
            if (m_opaque_sp->GetImages().ResolveFileAddress (start_addr, range.GetBaseAddress()))
                range.GetBaseAddress().SetOffset(start_addr);
        }

        // For now, we need a process;  the disassembly functions insist.  If we don't have one already,
        // make one.

        ExecutionContext exe_ctx;

        if (sb_process)
            sb_process->CalculateExecutionContext(exe_ctx);
        else 
            m_opaque_sp->CalculateExecutionContext(exe_ctx);

        if (end_addr == LLDB_INVALID_ADDRESS || end_addr < start_addr)
            range.SetByteSize( DEFAULT_DISASM_BYTE_SIZE);
        else
            range.SetByteSize(end_addr - start_addr);

        StreamFile out_stream (out);

        Disassembler::Disassemble (m_opaque_sp->GetDebugger(),
                                   m_opaque_sp->GetArchitecture(),
                                   exe_ctx,
                                   range,
                                   3,
                                   false,
                                   out_stream);
    }
}

void
SBTarget::Disassemble (const char *function_name, const char *module_name)
{
    if (function_name == NULL)
        return;
    
    FILE *out = m_opaque_sp->GetDebugger().GetOutputFileHandle();
    if (out == NULL)
        return;

    if (m_opaque_sp)
    {
        Disassembler *disassembler = Disassembler::FindPlugin (m_opaque_sp->GetArchitecture());
        if (disassembler == NULL)
          return;

        ModuleSP module_sp;
        if (module_name != NULL)
        {
            FileSpec module_file_spec (module_name, false);
            module_sp = m_opaque_sp->GetImages().FindFirstModuleForFileSpec (module_file_spec, NULL);
        }

        ExecutionContext exe_ctx;
        
        // Make sure the process object is alive if we have one (it might be
        // created but we might not be launched yet).
        Process *sb_process = m_opaque_sp->GetProcessSP().get();
        if (sb_process && !sb_process->IsAlive())
            sb_process = NULL;
        
        if (sb_process)
            sb_process->CalculateExecutionContext(exe_ctx);
        else 
            m_opaque_sp->CalculateExecutionContext(exe_ctx);


        StreamFile out_stream (out);

        Disassembler::Disassemble (m_opaque_sp->GetDebugger(),
                                   m_opaque_sp->GetArchitecture(),
                                   exe_ctx,
                                   ConstString (function_name),
                                   module_sp.get(),
                                   3,
                                   false,
                                   out_stream);
    }
}

bool
SBTarget::GetDescription (SBStream &description, lldb::DescriptionLevel description_level)
{
    if (m_opaque_sp)
    {
        description.ref();
        m_opaque_sp->Dump (description.get(), description_level);
    }
    else
        description.Printf ("No value");
    
    return true;
}

bool
SBTarget::GetDescription (SBStream &description, lldb::DescriptionLevel description_level) const
{
    if (m_opaque_sp)
    {
        description.ref();
        m_opaque_sp->Dump (description.get(), description_level);
    }
    else
        description.Printf ("No value");
    
    return true;
}
