//===-- DisassemblerLLVMC.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DisassemblerLLVMC.h"

#include "llvm-c/Disassembler.h"
#include "llvm/Support/TargetSelect.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/StackFrame.h"

#include <regex.h>

using namespace lldb;
using namespace lldb_private;

class InstructionLLVMC : public lldb_private::Instruction
{
public:
    InstructionLLVMC (DisassemblerLLVMC &disasm,
                      const lldb_private::Address &address, 
                      AddressClass addr_class) :
        Instruction(address, addr_class),
        m_is_valid(false),
        m_disasm(disasm),
        m_no_comments(true),
        m_comment_stream(),
        m_does_branch(eLazyBoolCalculate)
    {
    }
    
    virtual
    ~InstructionLLVMC ()
    {
    }
    
    static void
    PadToWidth (lldb_private::StreamString &ss,
                int new_width)
    {
        int old_width = ss.GetSize();
        
        if (old_width < new_width)
        {
            ss.Printf("%*s", new_width - old_width, "");
        }
    }
    
    virtual void
    Dump (lldb_private::Stream *s,
          uint32_t max_opcode_byte_size,
          bool show_address,
          bool show_bytes,
          const lldb_private::ExecutionContext* exe_ctx,
          bool raw)
    {
        const size_t opcode_column_width = 7;
        const size_t operand_column_width = 25;
             
        StreamString ss;
        
        ExecutionContextScope *exe_scope = NULL;
        
        if ((!raw) && exe_ctx)
        {
            exe_scope = exe_ctx->GetBestExecutionContextScope();

            DataExtractor extractor(m_raw_bytes.data(),
                                    m_raw_bytes.size(),
                                    m_disasm.GetArchitecture().GetByteOrder(),
                                    m_disasm.GetArchitecture().GetAddressByteSize());
            
            Parse <true> (m_address,
                          m_address_class,
                          extractor,
                          0,
                          exe_scope);
        }
        
        if (show_address)
        {
            m_address.Dump(&ss,
                           exe_scope,
                           Address::DumpStyleLoadAddress,
                           Address::DumpStyleModuleWithFileAddress,
                           0);
            
            ss.PutCString(":  ");
        }
        
        if (show_bytes)
        {
            if (m_opcode.GetType() == Opcode::eTypeBytes)
            {
                // x86_64 and i386 are the only ones that use bytes right now so
                // pad out the byte dump to be able to always show 15 bytes (3 chars each) 
                // plus a space
                if (max_opcode_byte_size > 0)
                    m_opcode.Dump (&ss, max_opcode_byte_size * 3 + 1);
                else
                    m_opcode.Dump (&ss, 15 * 3 + 1);
            }
            else
            {
                // Else, we have ARM which can show up to a uint32_t 0x00000000 (10 spaces)
                // plus two for padding...
                if (max_opcode_byte_size > 0)
                    m_opcode.Dump (&ss, max_opcode_byte_size * 3 + 1);
                else
                    m_opcode.Dump (&ss, 12);
            }        
        }
        
        int size_before_inst = ss.GetSize();
        
        ss.PutCString(m_opcode_name.c_str());
        
        PadToWidth(ss, size_before_inst + opcode_column_width);
        
        ss.PutCString(m_mnemocics.c_str());
        
        PadToWidth(ss, size_before_inst + opcode_column_width + operand_column_width);
        
        if (!m_comment.empty())
        {
            ss.PutCString(" ; ");
            ss.PutCString(m_comment.c_str());
        }
        
        ss.Flush();
        
        s->PutCString(ss.GetData());
    }
    
    virtual bool
    DoesBranch () const
    {
        return m_does_branch == eLazyBoolYes;
    }
    
    virtual size_t
    Decode (const lldb_private::Disassembler &disassembler,
            const lldb_private::DataExtractor &data,
            uint32_t data_offset)
    {
        Parse <false> (m_address, 
                       m_address_class, 
                       data,
                       data_offset,
                       NULL);
        
        return m_opcode.GetByteSize();
    }
    
    void
    AddReferencedAddress (std::string &description)
    {
        if (m_no_comments)
            m_comment_stream.PutCString(", ");
        else
            m_no_comments = true;
        
        m_comment_stream.PutCString(description.c_str());
    }
    
    virtual void
    CalculateMnemonicOperandsAndComment (lldb_private::ExecutionContextScope *exe_scope)
    {
        DataExtractor extractor(m_raw_bytes.data(),
                                m_raw_bytes.size(),
                                m_disasm.GetArchitecture().GetByteOrder(),
                                m_disasm.GetArchitecture().GetAddressByteSize());
        
        Parse <true> (m_address,
                      m_address_class,
                      extractor,
                      0,
                      exe_scope);
    }
    
    bool
    IsValid ()
    {
        return m_is_valid;
    }
    
    size_t
    GetByteSize ()
    {
        return m_opcode.GetByteSize();
    }
protected:
    void PopulateOpcode (const DataExtractor &extractor,
                         uint32_t offset,
                         size_t inst_size)
    {
        const ArchSpec &arch = m_disasm.GetArchitecture();
        llvm::Triple::ArchType machine = arch.GetMachine();
        
        switch (machine)
        {
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
            m_opcode.SetOpcodeBytes(extractor.PeekData(offset, inst_size), inst_size);
            return;

        case llvm::Triple::arm:
        case llvm::Triple::thumb:
            switch (inst_size)
            {
                case 2:
                    m_opcode.SetOpcode16 (extractor.GetU16 (&offset));
                    break;
                case 4:
                    if (machine == llvm::Triple::arm && m_address_class == eAddressClassCodeAlternateISA)
                    {
                        // If it is a 32-bit THUMB instruction, we need to swap the upper & lower halves.
                        uint32_t orig_bytes = extractor.GetU32 (&offset);
                        uint16_t upper_bits = (orig_bytes >> 16) & ((1u << 16) - 1);
                        uint16_t lower_bits = orig_bytes & ((1u << 16) - 1);
                        uint32_t swapped = (lower_bits << 16) | upper_bits;
                        m_opcode.SetOpcode32 (swapped);
                    }
                    else
                    {
                        m_opcode.SetOpcode32 (extractor.GetU32 (&offset));
                    }
                    break;
                default:
                    assert (!"Invalid ARM opcode size");
                    break;
            }
            return;

        default:
            break;
        }
        // Handle the default cases here.
        const uint32_t min_op_byte_size = arch.GetMinimumOpcodeByteSize();
        const uint32_t max_op_byte_size = arch.GetMaximumOpcodeByteSize();
        if (min_op_byte_size == max_op_byte_size)
        {
            assert (inst_size == min_op_byte_size);
            switch (inst_size)
            {
                case 1: m_opcode.SetOpcode8  (extractor.GetU8  (&offset)); return;
                case 2: m_opcode.SetOpcode16 (extractor.GetU16 (&offset)); return;
                case 4: m_opcode.SetOpcode32 (extractor.GetU32 (&offset)); return;
                case 8: m_opcode.SetOpcode64 (extractor.GetU64 (&offset)); return;
                default:
                    break;
            }
        }
        m_opcode.SetOpcodeBytes(extractor.PeekData(offset, inst_size), inst_size);
    }
    
    bool StringRepresentsBranch (const char *data, size_t size)
    {
        const char *cursor = data;

        bool inWhitespace = true;

        while (inWhitespace && cursor < data + size)
        {
            switch (*cursor)
            {
            default:
                inWhitespace = false;
                break;
            case ' ':
                break;
            case '\t':
                break;
            }
            
            if (inWhitespace)
                ++cursor;
        }
        
        if (cursor >= data + size)
            return false;
        
        llvm::Triple::ArchType arch = m_disasm.GetArchitecture().GetMachine();
        
        switch (arch)
        {
        default:
            return false;
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
            switch (cursor[0])
            {
            default:
                return false;
            case 'j':
                return true;
            case 'c':
                if (cursor[1] == 'a' &&
                    cursor[2] == 'l' &&
                    cursor[3] == 'l')
                    return true;
                else
                    return false;
            }
        case llvm::Triple::arm:
        case llvm::Triple::thumb:
            switch (cursor[0])
            {
            default:
                return false;
            case 'b':
                {
                    switch (cursor[1])
                    {
                    default:
                        return true;
                    case 'f':
                    case 'i':
                    case 'k':
                        return false;
                    }
                }
            case 'c':
                {
                    switch (cursor[1])
                    {
                    default:
                        return false;
                    case 'b':
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
    
    template <bool Reparse> bool Parse (const lldb_private::Address &address, 
                                        AddressClass addr_class,
                                        const DataExtractor &extractor,
                                        uint32_t data_offset,
                                        lldb_private::ExecutionContextScope *exe_scope)
    {
        std::vector<char> out_string(256);
        
        const uint8_t *data_start = extractor.GetDataStart();
        
        m_disasm.Lock(this, exe_scope);
        
        ::LLVMDisasmContextRef disasm_context;
        
        if (addr_class == eAddressClassCodeAlternateISA)
            disasm_context = m_disasm.m_alternate_disasm_context;
        else
            disasm_context = m_disasm.m_disasm_context;
        
        m_comment_stream.Clear();
        
        lldb::addr_t pc = LLDB_INVALID_ADDRESS;
        
        if (exe_scope)
            if (TargetSP target_sp = exe_scope->CalculateTarget())
                pc = m_address.GetLoadAddress(target_sp.get());
        
        if (pc == LLDB_INVALID_ADDRESS)
            pc = m_address.GetFileAddress();
                              
        size_t inst_size = ::LLVMDisasmInstruction(disasm_context,
                                                   const_cast<uint8_t*>(data_start) + data_offset,
                                                   extractor.GetByteSize() - data_offset,
                                                   pc, 
                                                   out_string.data(), 
                                                   out_string.size());
        
        if (m_does_branch == eLazyBoolCalculate)
            m_does_branch = (StringRepresentsBranch (out_string.data(), out_string.size()) ?
                             eLazyBoolYes : eLazyBoolNo);
        
        m_comment_stream.Flush();
        m_no_comments = false;
        
        m_comment.swap(m_comment_stream.GetString());

        m_disasm.Unlock();
        
        if (Reparse)
        {
            if (inst_size != m_raw_bytes.size())
                return false;
        }
        else
        {
            if (!inst_size)
                return false;
        }
            
        PopulateOpcode(extractor, data_offset, inst_size);
        
        m_raw_bytes.resize(inst_size);
        memcpy(m_raw_bytes.data(), data_start + data_offset, inst_size);
        
        if (!s_regex_compiled)
        {
            ::regcomp(&s_regex, "[ \t]*([^ ^\t]+)[ \t]*([^ ^\t].*)?", REG_EXTENDED);
            s_regex_compiled = true;
        }
        
        ::regmatch_t matches[3];
        
        const char *out_data = out_string.data();
        
        if (!::regexec(&s_regex, out_data, sizeof(matches) / sizeof(::regmatch_t), matches, 0))
        {
            if (matches[1].rm_so != -1)
                m_opcode_name.assign(out_data + matches[1].rm_so, matches[1].rm_eo - matches[1].rm_so);
            if (matches[2].rm_so != -1)
                m_mnemocics.assign(out_data + matches[2].rm_so, matches[2].rm_eo - matches[2].rm_so);
        }
                    
        m_is_valid = true;
    
        return true;
    }
                 
    bool                    m_is_valid;
    DisassemblerLLVMC      &m_disasm;
    std::vector<uint8_t>    m_raw_bytes;
    
    bool                    m_no_comments;
    StreamString            m_comment_stream;
    LazyBool                m_does_branch;
    
    static bool             s_regex_compiled;
    static ::regex_t        s_regex;
};

bool InstructionLLVMC::s_regex_compiled = false;
::regex_t InstructionLLVMC::s_regex;

Disassembler *
DisassemblerLLVMC::CreateInstance (const ArchSpec &arch)
{
    std::auto_ptr<DisassemblerLLVMC> disasm_ap (new DisassemblerLLVMC(arch));
    
    if (disasm_ap.get() && disasm_ap->IsValid())
        return disasm_ap.release();
    
    return NULL;
}

DisassemblerLLVMC::DisassemblerLLVMC (const ArchSpec &arch) :
    Disassembler(arch),
    m_disasm_context(NULL),
    m_alternate_disasm_context(NULL)
{
    m_disasm_context = ::LLVMCreateDisasm(arch.GetTriple().getTriple().c_str(), 
                                          (void*)this, 
                                          /*TagType=*/1,
                                          NULL,
                                          DisassemblerLLVMC::SymbolLookupCallback);
    
    if (arch.GetTriple().getArch() == llvm::Triple::arm)
    {
        m_alternate_disasm_context = ::LLVMCreateDisasm("thumbv7-apple-darwin", 
                                                        (void*)this, 
                                                        /*TagType=*/1,
                                                        NULL,
                                                        DisassemblerLLVMC::SymbolLookupCallback);
    }
}

DisassemblerLLVMC::~DisassemblerLLVMC()
{
    if (m_disasm_context)
    {
        ::LLVMDisasmDispose(m_disasm_context);
        m_disasm_context = NULL;
    }
    if (m_alternate_disasm_context)
    {
        ::LLVMDisasmDispose(m_alternate_disasm_context);
        m_alternate_disasm_context = NULL;
    }
}

size_t
DisassemblerLLVMC::DecodeInstructions (const Address &base_addr,
                                       const DataExtractor& data,
                                       uint32_t data_offset,
                                       uint32_t num_instructions,
                                       bool append)
{
    if (!append)
        m_instruction_list.Clear();
    
    if (!IsValid())
        return 0;
    
    uint32_t data_cursor = data_offset;
    size_t data_byte_size = data.GetByteSize();
    uint32_t instructions_parsed = 0;
    
    uint64_t instruction_pointer = base_addr.GetFileAddress();
        
    std::vector<char> out_string(256);
    
    while (data_offset < data_byte_size && instructions_parsed < num_instructions)
    {
        Address instr_address = base_addr;
        instr_address.Slide(data_cursor);
        
        AddressClass address_class = eAddressClassUnknown;
        
        if (m_alternate_disasm_context)
            address_class = instr_address.GetAddressClass ();
        
        InstructionSP inst_sp(new InstructionLLVMC(*this,
                                                   instr_address, 
                                                   address_class));
        
        if (!inst_sp)
            return data_cursor - data_offset;
            
        uint32_t inst_size = inst_sp->Decode(*this, data, data_cursor);
                
        if (!inst_size)
            return data_cursor - data_offset;
        
        m_instruction_list.Append(inst_sp);
        
        instruction_pointer += inst_size;
        data_cursor += inst_size;
        instructions_parsed++;
    }
    
    return data_cursor - data_offset;
}

void
DisassemblerLLVMC::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
    
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllDisassemblers();
}

void
DisassemblerLLVMC::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
DisassemblerLLVMC::GetPluginNameStatic()
{
    return "llvm-mc";
}

const char *
DisassemblerLLVMC::GetPluginDescriptionStatic()
{
    return "Disassembler that uses LLVM MC to disassemble i386, x86_64 and ARM.";
}

int DisassemblerLLVMC::OpInfoCallback (void *DisInfo,
                                       uint64_t PC,
                                       uint64_t Offset,
                                       uint64_t Size,
                                       int TagType,
                                       void *TagBug)
{
    return static_cast<DisassemblerLLVMC*>(DisInfo)->OpInfo(PC,
                                                            Offset,
                                                            Size,
                                                            TagType,
                                                            TagBug);
}

const char *DisassemblerLLVMC::SymbolLookupCallback(void *DisInfo,
                                                    uint64_t ReferenceValue,
                                                    uint64_t *ReferenceType,
                                                    uint64_t ReferencePC,
                                                    const char **ReferenceName)
{
    return static_cast<DisassemblerLLVMC*>(DisInfo)->SymbolLookup(ReferenceValue,
                                                                  ReferenceType,
                                                                  ReferencePC,
                                                                  ReferenceName);
}

int DisassemblerLLVMC::OpInfo (uint64_t PC,
                               uint64_t Offset,
                               uint64_t Size,
                               int TagType,
                               void *TagBug)
{
    switch (TagType)
    {
    default:
        break;
    case 1:
        bzero (TagBug, sizeof(::LLVMOpInfo1));
        break;
    }
    return 0;
}

const char *DisassemblerLLVMC::SymbolLookup (uint64_t ReferenceValue,
                                             uint64_t *ReferenceType,
                                             uint64_t ReferencePC,
                                             const char **ReferenceName)
{
    const char *result_name = NULL;
    uint64_t result_reference_type = LLVMDisassembler_ReferenceType_InOut_None;
    const char *result_referred_name = NULL;
    
    if (m_exe_scope && m_inst)
    {        
        Address reference_address;
        
        TargetSP target_sp (m_exe_scope->CalculateTarget());
        Target *target = target_sp.get();
        
        if (target)
        {
            if (!target->GetSectionLoadList().ResolveLoadAddress(ReferenceValue, reference_address))
            {
                if (ModuleSP module_sp = m_inst->GetAddress().GetModule())
                    module_sp->ResolveFileAddress(ReferenceValue, reference_address);
            }
            
            if (reference_address.IsValid() && reference_address.GetSection())
            {
                StreamString ss;
                
                reference_address.Dump (&ss, 
                                        target, 
                                        Address::DumpStyleResolvedDescriptionNoModule, 
                                        Address::DumpStyleSectionNameOffset);
                
                if (!ss.GetString().empty())
                    m_inst->AddReferencedAddress(ss.GetString());
            }
        }
    }
        
    *ReferenceType = result_reference_type;
    *ReferenceName = result_referred_name;
        
    return result_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
DisassemblerLLVMC::GetPluginName()
{
    return "DisassemblerLLVMC";
}

const char *
DisassemblerLLVMC::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
DisassemblerLLVMC::GetPluginVersion()
{
    return 1;
}

