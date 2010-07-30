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
}

SBTarget::SBTarget (const SBTarget& rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

SBTarget::SBTarget(const TargetSP& target_sp) :
    m_opaque_sp (target_sp)
{
}

const SBTarget&
SBTarget::Assign (const SBTarget& rhs)
{
    if (this != &rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
    }
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
    SBProcess sb_process;
    if (m_opaque_sp)
        sb_process.SetProcess (m_opaque_sp->GetProcessSP());
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

SBProcess
SBTarget::CreateProcess ()
{
    SBProcess sb_process;

    if (m_opaque_sp)
        sb_process.SetProcess (m_opaque_sp->CreateProcess (m_opaque_sp->GetDebugger().GetListener()));

    return sb_process;
}

SBProcess
SBTarget::LaunchProcess
(
    char const **argv,
    char const **envp,
    const char *tty,
    bool stop_at_entry
)
{
    SBProcess process(GetProcess ());
    if (!process.IsValid())
        process = CreateProcess();
    if (process.IsValid())
    {
        Error error (process->Launch (argv, envp, tty, tty, tty));
        if (error.Success())
        {
            if (!stop_at_entry)
            {
                StateType state = process->WaitForProcessToStop (NULL);
                if (state == eStateStopped)
                    process->Resume();
            }
        }
    }
    return process;
}

SBFileSpec
SBTarget::GetExecutable ()
{
    SBFileSpec exe_file_spec;
    if (m_opaque_sp)
    {
        ModuleSP exe_module_sp (m_opaque_sp->GetExecutableModule ());
        if (exe_module_sp)
            exe_file_spec.SetFileSpec (exe_module_sp->GetFileSpec());
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
SBTarget::MakeCurrentTarget ()
{
    if (m_opaque_sp)
    {
        m_opaque_sp->GetDebugger().GetTargetList().SetCurrentTarget (m_opaque_sp.get());
        return true;
    }
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
    SBBreakpoint sb_bp;
    if (file != NULL && line != 0)
        sb_bp = BreakpointCreateByLocation (SBFileSpec (file), line);
    return sb_bp;
}

SBBreakpoint
SBTarget::BreakpointCreateByLocation (const SBFileSpec &sb_file_spec, uint32_t line)
{
    SBBreakpoint sb_bp;
    if (m_opaque_sp.get() && line != 0)
        *sb_bp = m_opaque_sp->CreateBreakpoint (NULL, *sb_file_spec, line, true, false);
    return sb_bp;
}

SBBreakpoint
SBTarget::BreakpointCreateByName (const char *symbol_name, const char *module_name)
{
    SBBreakpoint sb_bp;
    if (m_opaque_sp.get() && symbol_name && symbol_name[0])
    {
        if (module_name && module_name[0])
        {
            FileSpec module_file_spec(module_name);
            *sb_bp = m_opaque_sp->CreateBreakpoint (&module_file_spec, symbol_name, false);
        }
        else
        {
            *sb_bp = m_opaque_sp->CreateBreakpoint (NULL, symbol_name, false);
        }
    }
    return sb_bp;
}

SBBreakpoint
SBTarget::BreakpointCreateByRegex (const char *symbol_name_regex, const char *module_name)
{
    SBBreakpoint sb_bp;
    if (m_opaque_sp.get() && symbol_name_regex && symbol_name_regex[0])
    {
        RegularExpression regexp(symbol_name_regex);
        
        if (module_name && module_name[0])
        {
            FileSpec module_file_spec(module_name);
            
            *sb_bp = m_opaque_sp->CreateBreakpoint (&module_file_spec, regexp, false);
        }
        else
        {
            *sb_bp = m_opaque_sp->CreateBreakpoint (NULL, regexp, false);
        }
    }
    return sb_bp;
}



SBBreakpoint
SBTarget::BreakpointCreateByAddress (addr_t address)
{
    SBBreakpoint sb_bp;
    if (m_opaque_sp.get())
        *sb_bp = m_opaque_sp->CreateBreakpoint (address, false);
    return sb_bp;
}

SBBreakpoint
SBTarget::FindBreakpointByID (break_id_t bp_id)
{
    SBBreakpoint sb_breakpoint;
    if (m_opaque_sp && bp_id != LLDB_INVALID_BREAK_ID)
        *sb_breakpoint = m_opaque_sp->GetBreakpointByID (bp_id);
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
    if (m_opaque_sp)
        return m_opaque_sp->RemoveBreakpointByID (bp_id);
    return false;
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
    if (m_opaque_sp)
        return m_opaque_sp->GetImages().GetSize();
    return 0;
}

void
SBTarget::Clear ()
{
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
    SBModule sb_module;
    if (m_opaque_sp)
        sb_module.SetModule(m_opaque_sp->GetImages().GetModuleAtIndex(idx));
    return sb_module;
}


SBBroadcaster
SBTarget::GetBroadcaster () const
{
    SBBroadcaster broadcaster(m_opaque_sp.get(), false);
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
            FileSpec module_file_spec (module_name);
            module_sp = m_opaque_sp->GetImages().FindFirstModuleForFileSpec (module_file_spec, NULL);
        }
        
        AddressRange range;

        // Make sure the process object is alive if we have one (it might be
        // created but we might not be launched yet).
        Process *process = m_opaque_sp->GetProcessSP().get();
        if (process && !process->IsAlive())
            process = NULL;
        
        // If we are given a module, then "start_addr" is a file address in
        // that module.
        if (module_sp)
        {
            if (!module_sp->ResolveFileAddress (start_addr, range.GetBaseAddress()))
                range.GetBaseAddress().SetOffset(start_addr);
        }
        else if (process)
        {
            // We don't have a module, se we need to figure out if "start_addr"
            // resolves to anything in a running process.
            if (!process->ResolveLoadAddress(start_addr, range.GetBaseAddress()))
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

        if (process)
            process->Calculate(exe_ctx);
        else 
            m_opaque_sp->Calculate(exe_ctx);

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
            FileSpec module_file_spec (module_name);
            module_sp = m_opaque_sp->GetImages().FindFirstModuleForFileSpec (module_file_spec, NULL);
        }

        ExecutionContext exe_ctx;
        
        // Make sure the process object is alive if we have one (it might be
        // created but we might not be launched yet).
        Process *process = m_opaque_sp->GetProcessSP().get();
        if (process && !process->IsAlive())
            process = NULL;
        
        if (process)
            process->Calculate(exe_ctx);
        else 
            m_opaque_sp->Calculate(exe_ctx);


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
