//===-- DisassemblerLLVM.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DisassemblerLLVM.h"

#include "llvm-c/EnhancedDisassembly.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/SymbolContext.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"

#include <assert.h>

using namespace lldb;
using namespace lldb_private;


static int 
DataExtractorByteReader (uint8_t *byte, uint64_t address, void *arg)
{
    DataExtractor &extractor = *((DataExtractor *)arg);

    if (extractor.ValidOffset(address))
    {
        *byte = *(extractor.GetDataStart() + address);
        return 0;
    }
    else
    {
        return -1;
    }
}

namespace {
    struct RegisterReaderArg {
        const lldb::addr_t instructionPointer;
        const EDDisassemblerRef disassembler;

        RegisterReaderArg(lldb::addr_t ip,
                          EDDisassemblerRef dis) :
            instructionPointer(ip),
            disassembler(dis)
        {
        }
    };
}

static int IPRegisterReader(uint64_t *value, unsigned regID, void* arg)
{
    uint64_t instructionPointer = ((RegisterReaderArg*)arg)->instructionPointer;
    EDDisassemblerRef disassembler = ((RegisterReaderArg*)arg)->disassembler;

    if (EDRegisterIsProgramCounter(disassembler, regID)) {
        *value = instructionPointer;
        return 0;
    }

    return -1;
}

DisassemblerLLVM::InstructionLLVM::InstructionLLVM (const Address &addr, EDDisassemblerRef disassembler) :
    Instruction (addr),
    m_disassembler (disassembler)
{
}

DisassemblerLLVM::InstructionLLVM::~InstructionLLVM()
{
}

static void
PadString(Stream *s, const std::string &str, size_t width)
{
    int diff = width - str.length();

    if (diff > 0)
        s->Printf("%s%*.*s", str.c_str(), diff, diff, "");
    else
        s->Printf("%s ", str.c_str());
}

void
DisassemblerLLVM::InstructionLLVM::Dump
(
    Stream *s,
    bool show_address,
    const DataExtractor *bytes,
    uint32_t bytes_offset,
    const lldb_private::ExecutionContext* exe_ctx,
    bool raw
)
{
    const size_t opcodeColumnWidth = 7;
    const size_t operandColumnWidth = 25;

    ExecutionContextScope *exe_scope = NULL;
    if (exe_ctx)
        exe_scope = exe_ctx->GetBestExecutionContextScope();

    // If we have an address, print it out
    if (GetAddress().IsValid() && show_address)
    {
        if (GetAddress().Dump (s, 
                               exe_scope, 
                               Address::DumpStyleLoadAddress, 
                               Address::DumpStyleModuleWithFileAddress,
                               0))
            s->PutCString(":  ");
    }

    // If we are supposed to show bytes, "bytes" will be non-NULL.
    if (bytes)
    {
        uint32_t bytes_dumped = bytes->Dump(s, bytes_offset, eFormatBytes, 1, EDInstByteSize(m_inst), UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0) - bytes_offset;
        // Allow for 15 bytes of opcodes since this is the max for x86_64.
        // TOOD: We need to taylor this better for different architectures. For 
        // ARM we would want to show 16 bit opcodes for Thumb as properly byte
        // swapped uint16_t values, or 32 bit values swapped values for ARM.
        const uint32_t default_num_opcode_bytes = 15;
        if (bytes_dumped * 3 < (default_num_opcode_bytes*3))
        {
            uint32_t indent_level = (default_num_opcode_bytes*3) - (bytes_dumped * 3) + 1;
            s->Printf("%*.*s", indent_level, indent_level, "");
        }
    }

    int numTokens = EDNumTokens(m_inst);

    int currentOpIndex = -1;

    std::auto_ptr<RegisterReaderArg> rra;
    
    if (!raw)
    {
        addr_t base_addr = LLDB_INVALID_ADDRESS;
        if (exe_ctx && exe_ctx->target && !exe_ctx->target->GetSectionLoadList().IsEmpty())
            base_addr = GetAddress().GetLoadAddress (exe_ctx->target);
        if (base_addr == LLDB_INVALID_ADDRESS)
            base_addr = GetAddress().GetFileAddress ();
        
        rra.reset(new RegisterReaderArg(base_addr + EDInstByteSize(m_inst), m_disassembler));
    }

    bool printTokenized = false;

    if (numTokens != -1)
    {
        printTokenized = true;

        // Handle the opcode column.

        StreamString opcode;

        int tokenIndex = 0;

        EDTokenRef token;
        const char *tokenStr;

        if (EDGetToken(&token, m_inst, tokenIndex))
            printTokenized = false;

        if (!printTokenized || !EDTokenIsOpcode(token))
            printTokenized = false;

        if (!printTokenized || EDGetTokenString(&tokenStr, token))
            printTokenized = false;

        // Put the token string into our opcode string
        opcode.PutCString(tokenStr);

        // If anything follows, it probably starts with some whitespace.  Skip it.

        tokenIndex++;

        if (printTokenized && tokenIndex < numTokens)
        {
            if(!printTokenized || EDGetToken(&token, m_inst, tokenIndex))
                printTokenized = false;

            if(!printTokenized || !EDTokenIsWhitespace(token))
                printTokenized = false;
        }

        tokenIndex++;

        // Handle the operands and the comment.

        StreamString operands;
        StreamString comment;

        if (printTokenized)
        {
            bool show_token;

            for (; tokenIndex < numTokens; ++tokenIndex)
            {
                if (EDGetToken(&token, m_inst, tokenIndex))
                    return;

                if (raw)
                {
                    show_token = true;
                }
                else
                {
                    int operandIndex = EDOperandIndexForToken(token);

                    if (operandIndex >= 0)
                    {
                        if (operandIndex != currentOpIndex)
                        {
                            show_token = true;

                            currentOpIndex = operandIndex;
                            EDOperandRef operand;

                            if (!EDGetOperand(&operand, m_inst, currentOpIndex))
                            {
                                if (EDOperandIsMemory(operand))
                                {
                                    uint64_t operand_value;

                                    if (!EDEvaluateOperand(&operand_value, operand, IPRegisterReader, rra.get()))
                                    {
                                        if (EDInstIsBranch(m_inst))
                                        {
                                            operands.Printf("0x%llx ", operand_value);
                                            show_token = false;
                                        }
                                        else
                                        {
                                            // Put the address value into the comment
                                            comment.Printf("0x%llx ", operand_value);
                                        }

                                        lldb_private::Address so_addr;
                                        if (exe_ctx && exe_ctx->target && !exe_ctx->target->GetSectionLoadList().IsEmpty())
                                        {
                                            if (exe_ctx->target->GetSectionLoadList().ResolveLoadAddress (operand_value, so_addr))
                                                so_addr.Dump(&comment, exe_scope, Address::DumpStyleResolvedDescriptionNoModule, Address::DumpStyleSectionNameOffset);
                                        }
                                        else
                                        {
                                            Module *module = GetAddress().GetModule();
                                            if (module)
                                            {
                                                if (module->ResolveFileAddress (operand_value, so_addr))
                                                    so_addr.Dump(&comment, exe_scope, Address::DumpStyleResolvedDescriptionNoModule, Address::DumpStyleSectionNameOffset);
                                            }
                                        }

                                    } // EDEvaluateOperand
                                } // EDOperandIsMemory
                            } // EDGetOperand
                        } // operandIndex != currentOpIndex
                    } // operandIndex >= 0
                } // else(raw)

                if (show_token)
                {
                    if(EDGetTokenString(&tokenStr, token))
                    {
                        printTokenized = false;
                        break;
                    }

                    operands.PutCString(tokenStr);
                }
            } // for (tokenIndex)

            if (printTokenized)
            {
                if (operands.GetString().empty())
                {
                    s->PutCString(opcode.GetString().c_str());
                }
                else
                {
                    PadString(s, opcode.GetString(), opcodeColumnWidth);

                    if (comment.GetString().empty())
                    {
                        s->PutCString(operands.GetString().c_str());
                    }
                    else
                    {
                        PadString(s, operands.GetString(), operandColumnWidth);

                        s->PutCString("; ");
                        s->PutCString(comment.GetString().c_str());
                    } // else (comment.GetString().empty())
                } // else (operands.GetString().empty())
            } // printTokenized
        } // for (tokenIndex)
    } // numTokens != -1

    if (!printTokenized)
    {
        const char *str;

        if (EDGetInstString(&str, m_inst))
            return;
        else
            s->PutCString(str);
    }
}

bool
DisassemblerLLVM::InstructionLLVM::DoesBranch() const
{
    return EDInstIsBranch(m_inst);
}

size_t
DisassemblerLLVM::InstructionLLVM::GetByteSize() const
{
    return EDInstByteSize(m_inst);
}

size_t
DisassemblerLLVM::InstructionLLVM::Extract (const Disassembler &disassembler, 
                                            const lldb_private::DataExtractor &data,
                                            uint32_t data_offset)
{
    if (EDCreateInsts(&m_inst, 1, m_disassembler, DataExtractorByteReader, data_offset, (void*)(&data)))
    {
        const int byte_size = EDInstByteSize(m_inst);
        uint32_t offset = data_offset;
        // Make a copy of the opcode in m_opcode
        switch (disassembler.GetArchitecture().GetMachine())
        {
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
            m_opcode.SetOpcodeBytes (data.PeekData (data_offset, byte_size), byte_size);
            break;

        case llvm::Triple::arm:
            assert (byte_size == 4);
            m_opcode.SetOpcode32 (data.GetU32 (&offset));
            break;

        case llvm::Triple::thumb:
            assert ((byte_size == 2) || (byte_size == 4));
            if (byte_size == 2)
                m_opcode.SetOpcode16 (data.GetU16 (&offset));
            else
                m_opcode.SetOpcode32 (data.GetU32 (&offset));
            break;

        default:
            assert (!"This shouldn't happen since we control the architecture we allow DisassemblerLLVM to be created for");
            break;
        }
        return byte_size;
    }
    else
        return 0;
}

static inline EDAssemblySyntax_t
SyntaxForArchSpec (const ArchSpec &arch)
{
    switch (arch.GetMachine ())
    {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        return kEDAssemblySyntaxX86ATT;
    case llvm::Triple::arm:
        return kEDAssemblySyntaxARMUAL;
    default:
        break;
    }
    return (EDAssemblySyntax_t)0;   // default
}

Disassembler *
DisassemblerLLVM::CreateInstance(const ArchSpec &arch)
{
    std::auto_ptr<DisassemblerLLVM> disasm_ap (new DisassemblerLLVM(arch));
 
    if (disasm_ap->IsValid())
        return disasm_ap.release();

    return NULL;
}

DisassemblerLLVM::DisassemblerLLVM(const ArchSpec &arch) :
    Disassembler (arch),
    m_disassembler (NULL),
    m_disassembler_thumb (NULL) // For ARM only
{
    const std::string &arch_triple = arch.GetTriple().str();
    if (!arch_triple.empty())
    {
        if (EDGetDisassembler(&m_disassembler, arch_triple.c_str(), SyntaxForArchSpec (arch)))
            m_disassembler = NULL;
        llvm::Triple::ArchType llvm_arch = arch.GetTriple().getArch();
        if (llvm_arch == llvm::Triple::arm)
        {
            if (EDGetDisassembler(&m_disassembler_thumb, "thumb-apple-darwin", kEDAssemblySyntaxARMUAL))
                m_disassembler_thumb = NULL;
        }
        else if (llvm_arch == llvm::Triple::thumb)
        {
            m_disassembler_thumb = m_disassembler;
            if (EDGetDisassembler(&m_disassembler, "arm-apple-darwin-unknown", kEDAssemblySyntaxARMUAL))
                m_disassembler = NULL;
        }
    }
}

DisassemblerLLVM::~DisassemblerLLVM()
{
}

size_t
DisassemblerLLVM::DecodeInstructions
(
    const Address &base_addr,
    const DataExtractor& data,
    uint32_t data_offset,
    uint32_t num_instructions,
    bool append
)
{
    if (m_disassembler == NULL)
        return 0;

    size_t total_inst_byte_size = 0;

    if (!append)
        m_instruction_list.Clear();

    while (data.ValidOffset(data_offset) && num_instructions)
    {
        Address inst_addr (base_addr);
        inst_addr.Slide(data_offset);

        bool use_thumb = false;
        // If we have a thumb disassembler, then we have an ARM architecture
        // so we need to check what the instruction address class is to make
        // sure we shouldn't be disassembling as thumb...
        if (m_disassembler_thumb)
        {
            if (inst_addr.GetAddressClass () == eAddressClassCodeAlternateISA)
                use_thumb = true;
        }
        InstructionSP inst_sp (new InstructionLLVM (inst_addr, 
                                                    use_thumb ? m_disassembler_thumb : m_disassembler));

        size_t inst_byte_size = inst_sp->Extract (*this, data, data_offset);

        if (inst_byte_size == 0)
            break;

        m_instruction_list.Append (inst_sp);

        total_inst_byte_size += inst_byte_size;
        data_offset += inst_byte_size;
        num_instructions--;
    }

    return total_inst_byte_size;
}

void
DisassemblerLLVM::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
DisassemblerLLVM::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
DisassemblerLLVM::GetPluginNameStatic()
{
    return "disassembler.llvm";
}

const char *
DisassemblerLLVM::GetPluginDescriptionStatic()
{
    return "Disassembler that uses LLVM opcode tables to disassemble i386 and x86_64.";
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
DisassemblerLLVM::GetPluginName()
{
    return "DisassemblerLLVM";
}

const char *
DisassemblerLLVM::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
DisassemblerLLVM::GetPluginVersion()
{
    return 1;
}

