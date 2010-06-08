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
#include "lldb/Core/Args.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Options.h"
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
    m_load_addr()
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

    case 'a':
        m_load_addr = Args::StringToUInt64(optarg, LLDB_INVALID_ADDRESS, 0);
        if (m_load_addr == LLDB_INVALID_ADDRESS)
            m_load_addr = Args::StringToUInt64(optarg, LLDB_INVALID_ADDRESS, 16);

        if (m_load_addr == LLDB_INVALID_ADDRESS)
            error.SetErrorStringWithFormat ("Invalid address string '%s'.\n", optarg);
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
    m_load_addr = LLDB_INVALID_ADDRESS;
}

const lldb::OptionDefinition*
CommandObjectDisassemble::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}

lldb::OptionDefinition
CommandObjectDisassemble::CommandOptions::g_option_table[] =
{
{ 0, false, "bytes",    'b', no_argument,       NULL, 0, NULL,             "Show opcode bytes when disassembling."},
{ 0, false, "context",  'c', required_argument, NULL, 0, "<num-lines>",    "Number of context lines of source to show."},
{ 0, false, "mixed",    'm', no_argument,       NULL, 0, NULL,             "Enable mixed source and assembly display."},
{ 0, false, "raw",      'r', no_argument,       NULL, 0, NULL,             "Print raw disassembly with no symbol information."},

{ 1, false, "address",  'a', required_argument, NULL, 0, "<address>",      "Address to start disassembling."},
{ 1, false, "bytes",    'b', no_argument,       NULL, 0, NULL,             "Show opcode bytes when disassembling."},
{ 1, false, "context",  'c', required_argument, NULL, 0, "<num-lines>",    "Number of context lines of source to show."},
{ 1, false, "mixed",    'm', no_argument,       NULL, 0, NULL,             "Enable mixed source and assembly display."},
{ 1, false, "raw",      'r', no_argument,       NULL, 0, NULL,             "Print raw disassembly with no symbol information."},

{ 2, false, "name",     'n', required_argument, NULL, CommandCompletions::eSymbolCompletion, "<function-name>",             "Disassemble entire contents of the given function name."},
{ 2, false, "bytes",    'b', no_argument,       NULL, 0, NULL,             "Show opcode bytes when disassembling."},
{ 2, false, "context",  'c', required_argument, NULL, 0, "<num-lines>",    "Number of context lines of source to show."},
{ 2, false, "mixed",    'm', no_argument,       NULL, 0, NULL,             "Enable mixed source and assembly display."},
{ 2, false, "raw",      'r', no_argument,       NULL, 0, NULL,             "Print raw disassembly with no symbol information."},

{ 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};



//-------------------------------------------------------------------------
// CommandObjectDisassemble
//-------------------------------------------------------------------------

CommandObjectDisassemble::CommandObjectDisassemble () :
    CommandObject ("disassemble",
                     "Disassemble bytes in the current function or anywhere in the inferior program.",
                     "disassemble [[<start-addr> [<end-addr>]] | <function-name>] [<cmd-options>]")
{
}

CommandObjectDisassemble::~CommandObjectDisassemble()
{
}

void
CommandObjectDisassemble::Disassemble
(
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result,
    Disassembler *disassembler,
    const SymbolContextList &sc_list
)
{
    const size_t count = sc_list.GetSize();
    SymbolContext sc;
    AddressRange range;
    for (size_t i=0; i<count; ++i)
    {
        if (sc_list.GetContextAtIndex(i, sc) == false)
            break;
        if (sc.GetAddressRange(eSymbolContextFunction | eSymbolContextSymbol, range))
        {
            lldb::addr_t addr = range.GetBaseAddress().GetLoadAddress(context->GetExecutionContext().process);
            if (addr != LLDB_INVALID_ADDRESS)
            {
                lldb::addr_t end_addr = addr + range.GetByteSize();
                Disassemble (context, interpreter, result, disassembler, addr, end_addr);
            }
        }
    }
}

void
CommandObjectDisassemble::Disassemble
(
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result,
    Disassembler *disassembler,
    lldb::addr_t addr,
    lldb::addr_t end_addr
)
{
    if (addr == LLDB_INVALID_ADDRESS)
        return;

    if (end_addr == LLDB_INVALID_ADDRESS || addr >= end_addr)
        end_addr = addr + DEFAULT_DISASM_BYTE_SIZE;

    ExecutionContext exe_ctx (context->GetExecutionContext());
    DataExtractor data;
    size_t bytes_disassembled = disassembler->ParseInstructions (&exe_ctx, eAddressTypeLoad, addr, end_addr - addr, data);
    if (bytes_disassembled == 0)
    {
        // Nothing got disassembled...
    }
    else
    {
        // We got some things disassembled...
        size_t num_instructions = disassembler->GetInstructionList().GetSize();
        uint32_t offset = 0;
        Stream &output_stream = result.GetOutputStream();
        SymbolContext sc;
        SymbolContext prev_sc;
        AddressRange sc_range;
        if (m_options.show_mixed)
            output_stream.IndentMore ();

        for (size_t i=0; i<num_instructions; ++i)
        {
            Disassembler::Instruction *inst = disassembler->GetInstructionList().GetInstructionAtIndex (i);
            if (inst)
            {
                lldb::addr_t curr_addr = addr + offset;
                if (m_options.show_mixed)
                {
                    Process *process = context->GetExecutionContext().process;
                    if (!sc_range.ContainsLoadAddress (curr_addr, process))
                    {
                        prev_sc = sc;
                        Address curr_so_addr;
                        if (process && process->ResolveLoadAddress (curr_addr, curr_so_addr))
                        {
                            if (curr_so_addr.GetSection())
                            {
                                Module *module = curr_so_addr.GetSection()->GetModule();
                                uint32_t resolved_mask = module->ResolveSymbolContextForAddress(curr_so_addr, eSymbolContextEverything, sc);
                                if (resolved_mask)
                                {
                                    sc.GetAddressRange (eSymbolContextEverything, sc_range);
                                    if (sc != prev_sc)
                                    {
                                        if (offset != 0)
                                            output_stream.EOL();

                                        sc.DumpStopContext(&output_stream, process, curr_so_addr);
                                        output_stream.EOL();
                                        if (sc.comp_unit && sc.line_entry.IsValid())
                                        {
                                            interpreter->GetSourceManager().DisplaySourceLinesWithLineNumbers (
                                                    sc.line_entry.file,
                                                    sc.line_entry.line,
                                                    m_options.num_lines_context,
                                                    m_options.num_lines_context,
                                                    m_options.num_lines_context ? "->" : "",
                                                    &output_stream);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (m_options.show_mixed)
                    output_stream.IndentMore ();
                output_stream.Indent();
                size_t inst_byte_size = inst->GetByteSize();
                inst->Dump(&output_stream, curr_addr, m_options.show_bytes ? &data : NULL, offset, exe_ctx, m_options.raw);
                output_stream.EOL();
                offset += inst_byte_size;
                if (m_options.show_mixed)
                    output_stream.IndentLess ();
            }
            else
            {
                break;
            }
        }
        if (m_options.show_mixed)
            output_stream.IndentLess ();

    }
}

bool
CommandObjectDisassemble::Execute
(
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    Target *target = context->GetTarget();
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

    lldb::addr_t addr = LLDB_INVALID_ADDRESS;
    lldb::addr_t end_addr = LLDB_INVALID_ADDRESS;
    ConstString name;
    const size_t argc = command.GetArgumentCount();
    if (argc == 0 && m_options.m_load_addr != LLDB_INVALID_ADDRESS)
    {
        addr = m_options.m_load_addr;
        end_addr = addr + DEFAULT_DISASM_BYTE_SIZE;
    } else if (argc == 0 && !m_options.m_func_name.empty())
    {
        ConstString tmpname(m_options.m_func_name.c_str());
        name = tmpname;
    } else if (argc == 0)
    {
        ExecutionContext exe_ctx(context->GetExecutionContext());
        if (exe_ctx.frame)
        {
            SymbolContext sc(exe_ctx.frame->GetSymbolContext(eSymbolContextFunction | eSymbolContextSymbol));
            if (sc.function)
            {
                addr = sc.function->GetAddressRange().GetBaseAddress().GetLoadAddress(exe_ctx.process);
                if (addr != LLDB_INVALID_ADDRESS)
                    end_addr = addr + sc.function->GetAddressRange().GetByteSize();
            }
            else if (sc.symbol && sc.symbol->GetAddressRangePtr())
            {
                addr = sc.symbol->GetAddressRangePtr()->GetBaseAddress().GetLoadAddress(exe_ctx.process);
                if (addr != LLDB_INVALID_ADDRESS)
                {
                    end_addr = addr + sc.symbol->GetAddressRangePtr()->GetByteSize();
                    if (addr == end_addr)
                        end_addr += DEFAULT_DISASM_BYTE_SIZE;
                }
            }
            else
            {
                addr = exe_ctx.frame->GetPC().GetLoadAddress(exe_ctx.process);
                if (addr != LLDB_INVALID_ADDRESS)
                    end_addr = addr + DEFAULT_DISASM_BYTE_SIZE;
            }
        }
        else
        {
            result.AppendError ("invalid frame");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
    }
    else if (argc == 1)
    {
        const char *arg = command.GetArgumentAtIndex(0);
        addr = Args::StringToAddress (arg);
        if (addr == LLDB_INVALID_ADDRESS)
        {   
            // Lookup function or symbol name?
            ConstString tmpname(arg);
            name = tmpname;
        }
        else
        {
            end_addr = addr + DEFAULT_DISASM_BYTE_SIZE;
        }
    }
    else if (argc >= 1 && argc <= 2)
    {
        addr = Args::StringToAddress (command.GetArgumentAtIndex(0));
        if (addr == LLDB_INVALID_ADDRESS)
        {
            result.AppendErrorWithFormat ("Unable to parse address '%s'.\n", command.GetArgumentAtIndex(0));
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        end_addr = Args::StringToAddress (command.GetArgumentAtIndex(1), addr);
        if (end_addr == LLDB_INVALID_ADDRESS)
        {
            result.AppendErrorWithFormat ("Unable to parse address '%s'.\n", command.GetArgumentAtIndex(1));
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
    }

    if (!name.IsEmpty())
    {
        SymbolContextList sc_list;

        if (target->GetImages().FindFunctions(name, sc_list))
        {
            Disassemble (context, interpreter, result, disassembler, sc_list);
        }
        else if (target->GetImages().FindSymbolsWithNameAndType(name, eSymbolTypeCode, sc_list))
        {
            Disassemble (context, interpreter, result, disassembler, sc_list);
        }
        else
        {
            result.AppendErrorWithFormat ("Unable to find symbol with name '%s'.\n", name.GetCString());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
    }

    if (addr < end_addr)
    {
        Disassemble (context, interpreter, result, disassembler, addr, end_addr);
    }

    if (addr == LLDB_INVALID_ADDRESS && name.IsEmpty())
    {
        result.AppendError ("No recognizable address of function name provided");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
    {
        return result.Succeeded();
    }
}
