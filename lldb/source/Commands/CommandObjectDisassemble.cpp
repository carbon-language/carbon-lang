//===-- CommandObjectDisassemble.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectDisassemble.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/AddressRange.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#define DEFAULT_DISASM_BYTE_SIZE 32
#define DEFAULT_DISASM_NUM_INS  4

using namespace lldb;
using namespace lldb_private;

CommandObjectDisassemble::CommandOptions::CommandOptions () :
    Options(),
    num_lines_context(0),
    num_instructions (0),
    m_func_name(),
    m_start_addr(),
    m_end_addr (),
    m_at_pc (false)
{
    ResetOptionValues();
}

CommandObjectDisassemble::CommandOptions::~CommandOptions ()
{
}

Error
CommandObjectDisassemble::CommandOptions::SetOptionValue (int option_idx, const char *option_arg)
{
    Error error;

    char short_option = (char) m_getopt_table[option_idx].val;

    bool success;
    
    switch (short_option)
    {
    case 'm':
        show_mixed = true;
        break;

    case 'x':
        num_lines_context = Args::StringToUInt32(option_arg, 0, 0, &success);
        if (!success)
            error.SetErrorStringWithFormat ("Invalid num context lines string: \"%s\".\n", option_arg);
        break;

    case 'c':
        num_instructions = Args::StringToUInt32(option_arg, 0, 0, &success);
        if (!success)
            error.SetErrorStringWithFormat ("Invalid num of instructions string: \"%s\".\n", option_arg);
        break;

    case 'b':
        show_bytes = true;
        break;

    case 's':
        m_start_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS, 0);
        if (m_start_addr == LLDB_INVALID_ADDRESS)
            m_start_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS, 16);

        if (m_start_addr == LLDB_INVALID_ADDRESS)
            error.SetErrorStringWithFormat ("Invalid start address string '%s'.\n", option_arg);
        break;
    case 'e':
        m_end_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS, 0);
        if (m_end_addr == LLDB_INVALID_ADDRESS)
            m_end_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS, 16);

        if (m_end_addr == LLDB_INVALID_ADDRESS)
            error.SetErrorStringWithFormat ("Invalid end address string '%s'.\n", option_arg);
        break;

    case 'n':
        m_func_name = option_arg;
        break;

    case 'p':
        m_at_pc = true;
        break;

    case 'r':
        raw = true;
        break;

    case 'f':
        // The default action is to disassemble the function for the current frame.
        // There's no need to set any flag.
        break;

    default:
        error.SetErrorStringWithFormat("Unrecognized short option '%c'.\n", short_option);
        break;
    }

    return error;
}

void
CommandObjectDisassemble::CommandOptions::ResetOptionValues ()
{
    Options::ResetOptionValues();
    show_mixed = false;
    show_bytes = false;
    num_lines_context = 0;
    num_instructions = 0;
    m_func_name.clear();
    m_at_pc = false;
    m_start_addr = LLDB_INVALID_ADDRESS;
    m_end_addr = LLDB_INVALID_ADDRESS;
    raw = false;
}

const lldb::OptionDefinition*
CommandObjectDisassemble::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}

lldb::OptionDefinition
CommandObjectDisassemble::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_ALL, false, "bytes",    'b', no_argument,       NULL, 0, eArgTypeNone,             "Show opcode bytes when disassembling."},
{ LLDB_OPT_SET_ALL, false, "context",  'x', required_argument, NULL, 0, eArgTypeNumLines,    "Number of context lines of source to show."},
{ LLDB_OPT_SET_ALL, false, "mixed",    'm', no_argument,       NULL, 0, eArgTypeNone,             "Enable mixed source and assembly display."},
{ LLDB_OPT_SET_ALL, false, "raw",      'r', no_argument,       NULL, 0, eArgTypeNone,             "Print raw disassembly with no symbol information."},

{ LLDB_OPT_SET_1, true, "start-address",  's', required_argument, NULL, 0, eArgTypeStartAddress,      "Address at which to start disassembling."},
{ LLDB_OPT_SET_1, false, "end-address",  'e', required_argument, NULL, 0, eArgTypeEndAddress,      "Address at which to end disassembling."},

{ LLDB_OPT_SET_2, true, "start-address",  's', required_argument, NULL, 0, eArgTypeStartAddress,      "Address at which to start disassembling."},
{ LLDB_OPT_SET_2, false, "instruction-count",  'c', required_argument, NULL, 0, eArgTypeNumLines,      "Number of instructions to display."},

{ LLDB_OPT_SET_3, true, "name",     'n', required_argument, NULL, CommandCompletions::eSymbolCompletion, eArgTypeFunctionName,             "Disassemble entire contents of the given function name."},
{ LLDB_OPT_SET_3, false, "instruction-count",  'c', required_argument, NULL, 0, eArgTypeNumLines,      "Number of instructions to display."},

{ LLDB_OPT_SET_4, true, "current-frame", 'f',  no_argument, NULL, 0, eArgTypeNone,             "Disassemble from the start of the current frame's function."},
{ LLDB_OPT_SET_4, false, "instruction-count",  'c', required_argument, NULL, 0, eArgTypeNumLines,      "Number of instructions to display."},

{ LLDB_OPT_SET_5, true, "current-pc", 'p',  no_argument, NULL, 0, eArgTypeNone,             "Disassemble from the current pc."},
{ LLDB_OPT_SET_5, false, "instruction-count",  'c', required_argument, NULL, 0, eArgTypeNumLines,      "Number of instructions to display."},

{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};



//-------------------------------------------------------------------------
// CommandObjectDisassemble
//-------------------------------------------------------------------------

CommandObjectDisassemble::CommandObjectDisassemble (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "disassemble",
                   "Disassemble bytes in the current function, or elsewhere in the executable program as specified by the user.",
                   "disassemble [<cmd-options>]")
{
}

CommandObjectDisassemble::~CommandObjectDisassemble()
{
}

bool
CommandObjectDisassemble::Execute
(
    Args& command,
    CommandReturnObject &result
)
{
    Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
    if (target == NULL)
    {
        result.AppendError ("invalid target, set executable file using 'file' command");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    ArchSpec arch(target->GetArchitecture());
    if (!arch.IsValid())
    {
        result.AppendError ("target needs valid architecure in order to be able to disassemble");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    Disassembler *disassembler = Disassembler::FindPlugin(arch);

    if (disassembler == NULL)
    {
        result.AppendErrorWithFormat ("Unable to find Disassembler plug-in for %s architecture.\n", arch.GetArchitectureName());
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    result.SetStatus (eReturnStatusSuccessFinishResult);

    if (command.GetArgumentCount() != 0)
    {
        result.AppendErrorWithFormat ("\"disassemble\" arguments are specified as options.\n");
        GetOptions()->GenerateOptionUsage (m_interpreter,
                                           result.GetErrorStream(), 
                                           this);

        result.SetStatus (eReturnStatusFailed);
        return false;
    }
    
    if (m_options.show_mixed && m_options.num_lines_context == 0)
        m_options.num_lines_context = 1;

    ExecutionContext exe_ctx(m_interpreter.GetDebugger().GetExecutionContext());

    if (!m_options.m_func_name.empty())
    {
        ConstString name(m_options.m_func_name.c_str());
        
        if (Disassembler::Disassemble (m_interpreter.GetDebugger(), 
                                       arch,
                                       exe_ctx,
                                       name,
                                       NULL,    // Module *
                                       m_options.num_instructions,
                                       m_options.show_mixed ? m_options.num_lines_context : 0,
                                       m_options.show_bytes,
                                       m_options.raw,
                                       result.GetOutputStream()))
        {
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendErrorWithFormat ("Unable to find symbol with name '%s'.\n", name.GetCString());
            result.SetStatus (eReturnStatusFailed);
        }
    } 
    else
    {
        Address start_addr;
        lldb::addr_t range_byte_size = DEFAULT_DISASM_BYTE_SIZE;
        
        if (m_options.m_at_pc)
        {
            if (exe_ctx.frame == NULL)
            {
                result.AppendError ("Cannot disassemble around the current PC without a selected frame.\n");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            start_addr = exe_ctx.frame->GetFrameCodeAddress();
            if (m_options.num_instructions == 0)
            {
                // Disassembling at the PC always disassembles some number of instructions (not the whole function).
                m_options.num_instructions = DEFAULT_DISASM_NUM_INS;
            }
        }
        else
        {
            start_addr.SetOffset (m_options.m_start_addr);
            if (start_addr.IsValid())
            {
                if (m_options.m_end_addr != LLDB_INVALID_ADDRESS)
                {
                    if (m_options.m_end_addr < m_options.m_start_addr)
                    {
                        result.AppendErrorWithFormat ("End address before start address.\n");
                        result.SetStatus (eReturnStatusFailed);
                        return false;            
                    }
                    range_byte_size = m_options.m_end_addr - m_options.m_start_addr;
                }
            }
        }
        
        if (m_options.num_instructions != 0)
        {
            if (!start_addr.IsValid())
            {
                // The default action is to disassemble the current frame function.
                if (exe_ctx.frame)
                {
                    SymbolContext sc(exe_ctx.frame->GetSymbolContext(eSymbolContextFunction | eSymbolContextSymbol));
                    if (sc.function)
                        start_addr = sc.function->GetAddressRange().GetBaseAddress();
                    else if (sc.symbol && sc.symbol->GetAddressRangePtr())
                        start_addr = sc.symbol->GetAddressRangePtr()->GetBaseAddress();
                    else
                        start_addr = exe_ctx.frame->GetFrameCodeAddress();
                }
                
                if (!start_addr.IsValid())
                {
                    result.AppendError ("invalid frame");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }

            if (Disassembler::Disassemble (m_interpreter.GetDebugger(), 
                                           arch,
                                           exe_ctx,
                                           start_addr,
                                           m_options.num_instructions,
                                           m_options.show_mixed ? m_options.num_lines_context : 0,
                                           m_options.show_bytes,
                                           m_options.raw,
                                           result.GetOutputStream()))
            {
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
            else
            {
                result.AppendErrorWithFormat ("Failed to disassemble memory at 0x%8.8llx.\n", m_options.m_start_addr);
                result.SetStatus (eReturnStatusFailed);            
            }
        }
        else
        {
            AddressRange range;
            if (start_addr.IsValid())
            {
                range.GetBaseAddress() = start_addr;
                range.SetByteSize (range_byte_size);
            } 
            else
            {
                // The default action is to disassemble the current frame function.
                if (exe_ctx.frame)
                {
                    SymbolContext sc(exe_ctx.frame->GetSymbolContext(eSymbolContextFunction | eSymbolContextSymbol));
                    if (sc.function)
                        range = sc.function->GetAddressRange();
                    else if (sc.symbol && sc.symbol->GetAddressRangePtr())
                        range = *sc.symbol->GetAddressRangePtr();
                    else
                        range.GetBaseAddress() = exe_ctx.frame->GetFrameCodeAddress();
                }
                else
                {
                    result.AppendError ("invalid frame");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            if (range.GetByteSize() == 0)
                range.SetByteSize(DEFAULT_DISASM_BYTE_SIZE);

            if (Disassembler::Disassemble (m_interpreter.GetDebugger(), 
                                           arch,
                                           exe_ctx,
                                           range,
                                           m_options.num_instructions,
                                           m_options.show_mixed ? m_options.num_lines_context : 0,
                                           m_options.show_bytes,
                                           m_options.raw,
                                           result.GetOutputStream()))
            {
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
            else
            {
                result.AppendErrorWithFormat ("Failed to disassemble memory at 0x%8.8llx.\n", m_options.m_start_addr);
                result.SetStatus (eReturnStatusFailed);            
            }
        }
    }

    return result.Succeeded();
}
