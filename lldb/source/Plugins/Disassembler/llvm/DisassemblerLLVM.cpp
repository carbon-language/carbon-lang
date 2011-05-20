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

InstructionLLVM::InstructionLLVM (const Address &addr, 
                                  AddressClass addr_class,
                                  EDDisassemblerRef disassembler,
                                  llvm::Triple::ArchType arch_type) :
    Instruction (addr, addr_class),
    m_disassembler (disassembler),
    m_arch_type (arch_type)
{
}

InstructionLLVM::~InstructionLLVM()
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

#include "llvm/ADT/StringRef.h"
static void
StripSpaces(llvm::StringRef &Str)
{
    while (!Str.empty() && isspace(Str[0]))
        Str = Str.substr(1);
    while (!Str.empty() && isspace(Str.back()))
        Str = Str.substr(0, Str.size()-1);
}

void
InstructionLLVM::Dump
(
    Stream *s,
    uint32_t max_opcode_byte_size,
    bool show_address,
    bool show_bytes,
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
    if (show_bytes)
    {
        if (m_opcode.GetType() == Opcode::eTypeBytes)
        {
            // x86_64 and i386 are the only ones that use bytes right now so
            // pad out the byte dump to be able to always show 15 bytes (3 chars each) 
            // plus a space
            if (max_opcode_byte_size > 0)
                m_opcode.Dump (s, max_opcode_byte_size * 3 + 1);
            else
                m_opcode.Dump (s, 15 * 3 + 1);
        }
        else
        {
            // Else, we have ARM which can show up to a uint32_t 0x00000000 (10 spaces)
            // plus two for padding...
            if (max_opcode_byte_size > 0)
                m_opcode.Dump (s, max_opcode_byte_size * 3 + 1);
            else
                m_opcode.Dump (s, 12);
        }
    }

    int numTokens = -1;
    
    // FIXME!!!
    /* Remove the following section of code related to force_raw .... */
    bool force_raw = m_arch_type == llvm::Triple::arm ||
                     m_arch_type == llvm::Triple::thumb;
    if (!raw)
        raw = force_raw;
    /* .... when we fix the edis for arm/thumb. */

    if (!raw)
        numTokens = EDNumTokens(m_inst);

    int currentOpIndex = -1;

    bool printTokenized = false;

    if (numTokens != -1 && !raw)
    {
        addr_t base_addr = LLDB_INVALID_ADDRESS;
        
        if (exe_ctx && exe_ctx->target && !exe_ctx->target->GetSectionLoadList().IsEmpty())
            base_addr = GetAddress().GetLoadAddress (exe_ctx->target);
        if (base_addr == LLDB_INVALID_ADDRESS)
            base_addr = GetAddress().GetFileAddress ();
                    
        lldb::addr_t PC = base_addr + EDInstByteSize(m_inst);

        // When executing an ARM instruction, PC reads as the address of the
        // current instruction plus 8.  And for Thumb, it is plus 4.
        if (m_arch_type == llvm::Triple::arm)
          PC = base_addr + 8;
        else if (m_arch_type == llvm::Triple::thumb)
          PC = base_addr + 4;

        RegisterReaderArg rra(PC, m_disassembler);
        
        printTokenized = true;

        // Handle the opcode column.

        StreamString opcode;

        int tokenIndex = 0;

        EDTokenRef token;
        const char *tokenStr;

        if (EDGetToken(&token, m_inst, tokenIndex)) // 0 on success
            printTokenized = false;
        else if (!EDTokenIsOpcode(token))
            printTokenized = false;
        else if (EDGetTokenString(&tokenStr, token)) // 0 on success
            printTokenized = false;

        if (printTokenized)
        {
            // Put the token string into our opcode string
            opcode.PutCString(tokenStr);

            // If anything follows, it probably starts with some whitespace.  Skip it.
            if (++tokenIndex < numTokens)
            {
                if (EDGetToken(&token, m_inst, tokenIndex)) // 0 on success
                    printTokenized = false;
                else if (!EDTokenIsWhitespace(token))
                    printTokenized = false;
            }

            ++tokenIndex;
        }

        // Handle the operands and the comment.
        StreamString operands;
        StreamString comment;

        if (printTokenized)
        {
            bool show_token = false;

            for (; tokenIndex < numTokens; ++tokenIndex)
            {
                if (EDGetToken(&token, m_inst, tokenIndex))
                    return;

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

                                if (!EDEvaluateOperand(&operand_value, operand, IPRegisterReader, &rra))
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

            // FIXME!!!
            // Workaround for llvm::tB's operands not properly parsed by ARMAsmParser.
            if (m_arch_type == llvm::Triple::thumb && opcode.GetString() == "b") {
                const char *inst_str;
                const char *pos = NULL;
                if (EDGetInstString(&inst_str, m_inst) == 0 && (pos = strstr(inst_str, "#")) != NULL) {
                    uint64_t operand_value = PC + atoi(++pos);
                    operands.Printf("0x%llx ", operand_value);

                    lldb_private::Address so_addr;
                    if (exe_ctx && exe_ctx->target && !exe_ctx->target->GetSectionLoadList().IsEmpty()) {
                        if (exe_ctx->target->GetSectionLoadList().ResolveLoadAddress (operand_value, so_addr))
                            so_addr.Dump(&comment, exe_scope, Address::DumpStyleResolvedDescriptionNoModule, Address::DumpStyleSectionNameOffset);
                    } else {
                        Module *module = GetAddress().GetModule();
                        if (module) {
                            if (module->ResolveFileAddress (operand_value, so_addr))
                                so_addr.Dump(&comment, exe_scope, Address::DumpStyleResolvedDescriptionNoModule, Address::DumpStyleSectionNameOffset);
                        }
                    }
                }
            }
            // END of workaround.

            // If both operands and comment are empty, we will just print out
            // the raw disassembly.
            if (operands.GetString().empty() && comment.GetString().empty())
            {
                const char *str;

                if (EDGetInstString(&str, m_inst))
                    return;
                llvm::StringRef raw_disasm(str);
                StripSpaces(raw_disasm);
                s->PutCString(raw_disasm.str().c_str());
            }
            else
            {
                PadString(s, opcode.GetString(), opcodeColumnWidth);

                if (comment.GetString().empty())
                    s->PutCString(operands.GetString().c_str());
                else
                {
                    PadString(s, operands.GetString(), operandColumnWidth);

                    s->PutCString("; ");
                    s->PutCString(comment.GetString().c_str());
                } // else (comment.GetString().empty())
            } // else (operands.GetString().empty() && comment.GetString().empty())
        } // printTokenized
    } // numTokens != -1

    if (!printTokenized)
    {
        const char *str;

        if (EDGetInstString(&str, m_inst)) // 0 on success
            return;
        s->Write(str, strlen(str) - 1);
    }
}

bool
InstructionLLVM::DoesBranch() const
{
    return EDInstIsBranch(m_inst);
}

size_t
InstructionLLVM::Decode (const Disassembler &disassembler, 
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
        case llvm::Triple::thumb:
            switch (byte_size)
            {
            case 2: 
                m_opcode.SetOpcode16 (data.GetU16 (&offset)); 
                break;

            case 4:
                {
                if (GetAddressClass() ==  eAddressClassCodeAlternateISA)
                {
                    // If it is a 32-bit THUMB instruction, we need to swap the upper & lower halves.
                    uint32_t orig_bytes = data.GetU32 (&offset);
                    uint16_t upper_bits = (orig_bytes >> 16) & ((1u << 16) - 1);
                    uint16_t lower_bits = orig_bytes & ((1u << 16) - 1);
                    uint32_t swapped = (lower_bits << 16) | upper_bits;
                    m_opcode.SetOpcode32 (swapped);
                }
                else
                    m_opcode.SetOpcode32 (data.GetU32 (&offset));
                }
                break;

            default:
                assert (!"Invalid ARM opcode size");
                break;
            }
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
    case llvm::Triple::thumb:
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
 
    if (disasm_ap.get() && disasm_ap->IsValid())
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
		// Don't have the lldb::Triple::thumb architecture here. If someone specifies
		// "thumb" as the architecture, we want a thumb only disassembler. But if any
		// architecture starting with "arm" if specified, we want to auto detect the
		// arm/thumb code automatically using the AddressClass from section offset 
		// addresses.
        if (llvm_arch == llvm::Triple::arm)
        {
            if (EDGetDisassembler(&m_disassembler_thumb, "thumb-apple-darwin", kEDAssemblySyntaxARMUAL))
                m_disassembler_thumb = NULL;
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
        AddressClass inst_address_class = eAddressClassInvalid;
        if (m_disassembler_thumb)
        {
            inst_address_class = inst_addr.GetAddressClass ();
            if (inst_address_class == eAddressClassCodeAlternateISA)
                use_thumb = true;
        }
        
        InstructionSP inst_sp (new InstructionLLVM (inst_addr, 
                                                    inst_address_class,
                                                    use_thumb ? m_disassembler_thumb : m_disassembler,
                                                    use_thumb ? llvm::Triple::thumb : m_arch.GetMachine()));

        size_t inst_byte_size = inst_sp->Decode (*this, data, data_offset);

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
    return "llvm";
}

const char *
DisassemblerLLVM::GetPluginDescriptionStatic()
{
    return "Disassembler that uses LLVM opcode tables to disassemble i386, x86_64 and ARM.";
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

