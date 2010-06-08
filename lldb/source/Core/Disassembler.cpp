//===-- Disassembler.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Disassembler.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Timer.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

#define DEFAULT_DISASM_BYTE_SIZE 32

using namespace lldb;
using namespace lldb_private;


Disassembler*
Disassembler::FindPlugin (const ArchSpec &arch)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "Disassembler::FindPlugin (arch = %s)",
                        arch.AsCString());

    std::auto_ptr<Disassembler> disassembler_ap;
    DisassemblerCreateInstance create_callback;
    for (uint32_t idx = 0; (create_callback = PluginManager::GetDisassemblerCreateCallbackAtIndex(idx)) != NULL; ++idx)
    {
        disassembler_ap.reset (create_callback(arch));

        if (disassembler_ap.get())
            return disassembler_ap.release();
    }
    return NULL;
}

bool
Disassembler::Disassemble
(
    const ArchSpec &arch,
    const ExecutionContext &exe_ctx,
    uint32_t mixed_context_lines,
    Stream &strm
)
{
    Disassembler *disassembler = Disassembler::FindPlugin(arch);

    if (disassembler)
    {
        lldb::addr_t addr = LLDB_INVALID_ADDRESS;
        size_t byte_size = 0;
        if (exe_ctx.frame)
        {
            SymbolContext sc(exe_ctx.frame->GetSymbolContext(eSymbolContextFunction | eSymbolContextSymbol));
            if (sc.function)
            {
                addr = sc.function->GetAddressRange().GetBaseAddress().GetLoadAddress(exe_ctx.process);
                if (addr != LLDB_INVALID_ADDRESS)
                    byte_size = sc.function->GetAddressRange().GetByteSize();
            }
            else if (sc.symbol && sc.symbol->GetAddressRangePtr())
            {
                addr = sc.symbol->GetAddressRangePtr()->GetBaseAddress().GetLoadAddress(exe_ctx.process);
                if (addr != LLDB_INVALID_ADDRESS)
                {
                    byte_size = sc.symbol->GetAddressRangePtr()->GetByteSize();
                    if (byte_size == 0)
                        byte_size = DEFAULT_DISASM_BYTE_SIZE;
                }
            }
            else
            {
                addr = exe_ctx.frame->GetPC().GetLoadAddress(exe_ctx.process);
                if (addr != LLDB_INVALID_ADDRESS)
                    byte_size = DEFAULT_DISASM_BYTE_SIZE;
            }
        }

        if (byte_size)
        {
            DataExtractor data;
            size_t bytes_disassembled = disassembler->ParseInstructions (&exe_ctx, eAddressTypeLoad, addr, byte_size, data);
            if (bytes_disassembled == 0)
            {
                return false;
            }
            else
            {
                // We got some things disassembled...
                size_t num_instructions = disassembler->GetInstructionList().GetSize();
                uint32_t offset = 0;
                SymbolContext sc;
                SymbolContext prev_sc;
                AddressRange sc_range;
                if (mixed_context_lines)
                    strm.IndentMore ();

                for (size_t i=0; i<num_instructions; ++i)
                {
                    Disassembler::Instruction *inst = disassembler->GetInstructionList().GetInstructionAtIndex (i);
                    if (inst)
                    {
                        lldb::addr_t curr_addr = addr + offset;
                        if (mixed_context_lines)
                        {
                            if (!sc_range.ContainsLoadAddress (curr_addr, exe_ctx.process))
                            {
                                prev_sc = sc;
                                Address curr_so_addr;
                                Process *process = exe_ctx.process;
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
                                                    strm.EOL();

                                                sc.DumpStopContext(&strm, process, curr_so_addr);

                                                if (sc.comp_unit && sc.line_entry.IsValid())
                                                {
                                                    Debugger::GetSharedInstance().GetSourceManager().DisplaySourceLinesWithLineNumbers (
                                                            sc.line_entry.file,
                                                            sc.line_entry.line,
                                                            mixed_context_lines,
                                                            mixed_context_lines,
                                                            mixed_context_lines ? "->" : "",
                                                            &strm);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (mixed_context_lines)
                            strm.IndentMore ();
                        strm.Indent();
                        size_t inst_byte_size = inst->GetByteSize();
                        //inst->Dump(&strm, curr_addr, &data, offset);  // Do dump opcode bytes
                        inst->Dump(&strm, curr_addr, NULL, offset, exe_ctx, false); // Don't dump opcode bytes
                        strm.EOL();
                        offset += inst_byte_size;
                        if (mixed_context_lines)
                            strm.IndentLess ();
                    }
                    else
                    {
                        break;
                    }
                }
                if (mixed_context_lines)
                    strm.IndentLess ();

            }
        }
        return true;
    }
    return false;
}

Disassembler::Instruction::Instruction()
{
}

Disassembler::Instruction::~Instruction()
{
}


Disassembler::InstructionList::InstructionList() :
    m_instructions()
{
}

Disassembler::InstructionList::~InstructionList()
{
}

size_t
Disassembler::InstructionList::GetSize() const
{
    return m_instructions.size();
}


Disassembler::Instruction *
Disassembler::InstructionList::GetInstructionAtIndex (uint32_t idx)
{
    if (idx < m_instructions.size())
        return m_instructions[idx].get();
    return NULL;
}

const Disassembler::Instruction *
Disassembler::InstructionList::GetInstructionAtIndex (uint32_t idx) const
{
    if (idx < m_instructions.size())
        return m_instructions[idx].get();
    return NULL;
}

void
Disassembler::InstructionList::Clear()
{
  m_instructions.clear();
}

void
Disassembler::InstructionList::AppendInstruction (Instruction::shared_ptr &inst_sp)
{
    if (inst_sp)
        m_instructions.push_back(inst_sp);
}


size_t
Disassembler::ParseInstructions
(
    const ExecutionContext *exe_ctx,
    lldb::AddressType addr_type,
    lldb::addr_t addr,
    size_t byte_size,
    DataExtractor& data
)
{
    Process *process = exe_ctx->process;

    if (process == NULL)
        return 0;

    DataBufferSP data_sp(new DataBufferHeap (byte_size, '\0'));

    Error error;
    if (process->GetTarget().ReadMemory (addr_type, addr, data_sp->GetBytes(), data_sp->GetByteSize(), error, NULL))
    {
        data.SetData(data_sp);
        data.SetByteOrder(process->GetByteOrder());
        data.SetAddressByteSize(process->GetAddressByteSize());
        return ParseInstructions (data, 0, UINT32_MAX, addr);
    }

    return 0;
}

//----------------------------------------------------------------------
// Disassembler copy constructor
//----------------------------------------------------------------------
Disassembler::Disassembler(const ArchSpec& arch) :
    m_arch (arch),
    m_instruction_list(),
    m_base_addr(LLDB_INVALID_ADDRESS)
{

}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Disassembler::~Disassembler()
{
}

Disassembler::InstructionList &
Disassembler::GetInstructionList ()
{
    return m_instruction_list;
}

const Disassembler::InstructionList &
Disassembler::GetInstructionList () const
{
    return m_instruction_list;
}
