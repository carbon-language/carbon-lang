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
    {
        SBListener sb_listener (m_opaque_sp->GetDebugger().GetListener());
        if (sb_listener.IsValid())
            sb_process.SetProcess (m_opaque_sp->CreateProcess (*sb_listener));
    }
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

void
SBTarget::ListAllBreakpoints ()
{
    FILE *out_file = m_opaque_sp->GetDebugger().GetOutputFileHandle();
    
    if (out_file == NULL)
        return;

    if (m_opaque_sp)
    {
        const BreakpointList &bp_list = m_opaque_sp->GetBreakpointList();
        size_t num_bps = bp_list.GetSize();
        for (int i = 0; i < num_bps; ++i)
        {
            SBBreakpoint sb_breakpoint (bp_list.GetBreakpointByIndex (i));
            sb_breakpoint.GetDescription (out_file, "full");
        }
    }
}

SBBreakpoint
SBTarget::FindBreakpointByID (break_id_t bp_id)
{
    SBBreakpoint sb_breakpoint;
    if (m_opaque_sp && bp_id != LLDB_INVALID_BREAK_ID)
        *sb_breakpoint = m_opaque_sp->GetBreakpointByID (bp_id);
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
SBTarget::Disassemble (lldb::addr_t file_address_start, lldb::addr_t file_address_end, const char *module_name)
{
    if (file_address_start == LLDB_INVALID_ADDRESS)
        return;

    FILE *out = m_opaque_sp->GetDebugger().GetOutputFileHandle();
    if (out == NULL)
        return;

    if (m_opaque_sp)
    {
        SBModule module;
        if (module_name != NULL)
        {
            SBFileSpec file_spec (module_name);
            module = FindModule (file_spec);
        }
        ArchSpec arch (m_opaque_sp->GetArchitecture());
        if (!arch.IsValid())
          return;
        Disassembler *disassembler = Disassembler::FindPlugin (arch);
        if (disassembler == NULL)
          return;

        // For now, we need a process;  the disassembly functions insist.  If we don't have one already,
        // make one.

        SBProcess process = GetProcess();
        if (! process.IsValid())
          process = CreateProcess();

        ExecutionContext exe_context (process.get());

        if (file_address_end == LLDB_INVALID_ADDRESS
            || file_address_end < file_address_start)
          file_address_end = file_address_start + DEFAULT_DISASM_BYTE_SIZE;

        // TO BE FIXED:  SOMEHOW WE NEED TO SPECIFY/USE THE MODULE, IF THE USER SPECIFIED ONE.  I'M NOT
        // SURE HOW TO DO THAT AT THE MOMENT.  WE ALSO NEED TO FIGURE OUT WHAT TO DO IF THERE ARE MULTIPLE
        // MODULES CONTAINING THE SPECIFIED ADDRESSES (E.G. THEY HAVEN'T ALL LOADED & BEEN GIVEN UNIQUE
        // ADDRESSES YET).

        DataExtractor data;
        size_t bytes_disassembled = disassembler->ParseInstructions (&exe_context, eAddressTypeLoad,
                                                                     file_address_start,
                                                                     file_address_end - file_address_start, data);

        if (bytes_disassembled > 0)
        {
            size_t num_instructions = disassembler->GetInstructionList().GetSize();
            uint32_t offset = 0;
            StreamFile out_stream (out);

            for (size_t i = 0; i < num_instructions; ++i)
            {
                Disassembler::Instruction *inst = disassembler->GetInstructionList().GetInstructionAtIndex (i);
                if (inst)
                {
                    lldb::addr_t cur_addr = file_address_start + offset;
                    size_t inst_byte_size = inst->GetByteSize();
                    inst->Dump (&out_stream, cur_addr, &data, offset, exe_context, false);
                    out_stream.EOL();
                    offset += inst_byte_size;
                }
            }
        }
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
        SBModule module;

        if (module_name != NULL)
        {
            SBFileSpec file_spec (module_name);
            module = FindModule (file_spec);
        }

        ArchSpec arch (m_opaque_sp->GetArchitecture());
        if (!arch.IsValid())
          return;

        Disassembler *disassembler = Disassembler::FindPlugin (arch);
        if (disassembler == NULL)
          return;

        // For now, we need a process;  the disassembly functions insist.  If we don't have one already,
        // make one.

        SBProcess process = GetProcess();
        if (! process.IsValid()
            ||  process.GetProcessID() == 0)
        {
            fprintf (out, "Cannot disassemble functions until after process has launched.\n");
            return;
        }

        ExecutionContext exe_context (process.get());

        FileSpec *containing_module = NULL;

        if (module_name != NULL)
            containing_module = new FileSpec (module_name);

        SearchFilterSP filter_sp (m_opaque_sp->GetSearchFilterForModule (containing_module));
        AddressResolverSP resolver_sp (new AddressResolverName (function_name));

        resolver_sp->ResolveAddress (*filter_sp);

        size_t num_matches_found = resolver_sp->GetNumberOfAddresses();

        if (num_matches_found == 1)
        {
            DataExtractor data;

            AddressRange func_addresses = resolver_sp->GetAddressRangeAtIndex (0);
            Address start_addr = func_addresses.GetBaseAddress();
            lldb::addr_t num_bytes = func_addresses.GetByteSize();

            lldb::addr_t addr = LLDB_INVALID_ADDRESS;
            size_t bytes_disassembled = 0;


            if (process.GetProcessID() == 0)
            {
                // Leave this branch in for now, but it should not be reached, since we exit above if the PID is 0.
                addr = start_addr.GetFileAddress ();
                bytes_disassembled = disassembler->ParseInstructions (&exe_context, eAddressTypeFile, addr,
                                                                      num_bytes, data);

            }
            else
            {
                addr = start_addr.GetLoadAddress (process.get());
                bytes_disassembled = disassembler->ParseInstructions (&exe_context, eAddressTypeLoad, addr,
                                                                      num_bytes, data);

            }

            if (bytes_disassembled > 0)
            {
                size_t num_instructions = disassembler->GetInstructionList().GetSize();
                uint32_t offset = 0;
                StreamFile out_stream (out);

                for (size_t i = 0; i < num_instructions; ++i)
                {
                    Disassembler::Instruction *inst = disassembler->GetInstructionList().GetInstructionAtIndex (i);
                    if (inst)
                    {
                        lldb::addr_t cur_addr = addr + offset;
                        size_t inst_byte_size = inst->GetByteSize();
                        inst->Dump (&out_stream, cur_addr, &data, offset, exe_context, false);
                        out_stream.EOL();
                        offset += inst_byte_size;
                    }
                }
            }
        }
        else if (num_matches_found > 1)
        {
            // TO BE FIXED:  Eventually we want to list/disassemble all functions found.
            fprintf (out, "Function '%s' was found in multiple modules; please specify the desired module name.\n",
                     function_name);
        }
        else
            fprintf (out, "Function '%s' was not found.\n", function_name);
    }
}
