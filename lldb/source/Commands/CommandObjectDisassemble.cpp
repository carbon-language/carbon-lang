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

using namespace lldb;
using namespace lldb_private;

CommandObjectDisassemble::CommandOptions::CommandOptions () :
    Options(),
    m_func_name(),
    m_start_addr(),
    m_end_addr ()
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

    switch (short_option)
    {
    case 'm':
        show_mixed = true;
        break;

    case 'c':
        num_lines_context = Args::StringToUInt32(option_arg, 0, 0);
        break;

    case 'b':
        show_bytes = true;
        break;

    case 's':
        m_start_addr = Args::StringToUInt64(optarg, LLDB_INVALID_ADDRESS, 0);
        if (m_start_addr == LLDB_INVALID_ADDRESS)
            m_start_addr = Args::StringToUInt64(optarg, LLDB_INVALID_ADDRESS, 16);

        if (m_start_addr == LLDB_INVALID_ADDRESS)
            error.SetErrorStringWithFormat ("Invalid start address string '%s'.\n", optarg);
        break;
    case 'e':
        m_end_addr = Args::StringToUInt64(optarg, LLDB_INVALID_ADDRESS, 0);
        if (m_end_addr == LLDB_INVALID_ADDRESS)
            m_end_addr = Args::StringToUInt64(optarg, LLDB_INVALID_ADDRESS, 16);

        if (m_end_addr == LLDB_INVALID_ADDRESS)
            error.SetErrorStringWithFormat ("Invalid end address string '%s'.\n", optarg);
        break;

    case 'n':
        m_func_name = option_arg;
        break;

    case 'r':
        raw = true;
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
    m_func_name.clear();
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
{ LLDB_OPT_SET_ALL, false, "bytes",    'b', no_argument,       NULL, 0, NULL,             "Show opcode bytes when disassembling."},
{ LLDB_OPT_SET_ALL, false, "context",  'c', required_argument, NULL, 0, "<num-lines>",    "Number of context lines of source to show."},
{ LLDB_OPT_SET_ALL, false, "mixed",    'm', no_argument,       NULL, 0, NULL,             "Enable mixed source and assembly display."},
{ LLDB_OPT_SET_ALL, false, "raw",      'r', no_argument,       NULL, 0, NULL,             "Print raw disassembly with no symbol information."},

{ LLDB_OPT_SET_1, true, "start-address",  's', required_argument, NULL, 0, "<start-address>",      "Address to start disassembling."},
{ LLDB_OPT_SET_1, false, "end-address",  'e', required_argument, NULL, 0, "<end-address>",      "Address to start disassembling."},

{ LLDB_OPT_SET_2, true, "name",     'n', required_argument, NULL, CommandCompletions::eSymbolCompletion, "<function-name>",             "Disassemble entire contents of the given function name."},

{ LLDB_OPT_SET_3, false, "current-frame",     'f', no_argument, NULL, 0, "<current-frame>",             "Disassemble entire contents of the current frame's function."},

{ 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};



//-------------------------------------------------------------------------
// CommandObjectDisassemble
//-------------------------------------------------------------------------

CommandObjectDisassemble::CommandObjectDisassemble () :
    CommandObject ("disassemble",
                     "Disassemble bytes in the current function or anywhere in the inferior program.",
                     "disassemble [<cmd-options>]")
{
}

CommandObjectDisassemble::~CommandObjectDisassemble()
{
}

bool
CommandObjectDisassemble::Execute
(
    CommandInterpreter &interpreter,
    Args& command,
    CommandReturnObject &result
)
{
    Target *target = interpreter.GetDebugger().GetCurrentTarget().get();
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
        result.AppendErrorWithFormat ("Unable to find Disassembler plug-in for %s architecture.\n", arch.AsCString());
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    result.SetStatus (eReturnStatusSuccessFinishResult);

    if (command.GetArgumentCount() != 0)
    {
        result.AppendErrorWithFormat ("\"disassemble\" doesn't take any arguments.\n");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
    ExecutionContext exe_ctx(interpreter.GetDebugger().GetExecutionContext());

    if (m_options.show_mixed && m_options.num_lines_context == 0)
        m_options.num_lines_context = 3;

    if (!m_options.m_func_name.empty())
    {
        ConstString name(m_options.m_func_name.c_str());
        
        if (Disassembler::Disassemble (interpreter.GetDebugger(), 
                                       arch,
                                       exe_ctx,
                                       name,
                                       NULL,    // Module *
                                       m_options.show_mixed ? m_options.num_lines_context : 0,
                                       m_options.show_bytes,
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
        AddressRange range;
        if (m_options.m_start_addr != LLDB_INVALID_ADDRESS)
        {
            range.GetBaseAddress().SetOffset (m_options.m_start_addr);
            if (m_options.m_end_addr != LLDB_INVALID_ADDRESS)
            {
                if (m_options.m_end_addr < m_options.m_start_addr)
                {
                    result.AppendErrorWithFormat ("End address before start address.\n");
                    result.SetStatus (eReturnStatusFailed);
                    return false;            
                }
                range.SetByteSize (m_options.m_end_addr - m_options.m_start_addr);
            }
            else
                range.SetByteSize (DEFAULT_DISASM_BYTE_SIZE);
        } 
        else
        {
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

        if (Disassembler::Disassemble (interpreter.GetDebugger(), 
                                       arch,
                                       exe_ctx,
                                       range,
                                       m_options.show_mixed ? m_options.num_lines_context : 0,
                                       m_options.show_bytes,
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

    return result.Succeeded();
}
