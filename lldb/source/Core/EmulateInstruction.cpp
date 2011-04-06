//===-- EmulateInstruction.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/EmulateInstruction.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Endian.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"

#include "Plugins/Instruction/ARM/EmulateInstructionARM.h"

using namespace lldb;
using namespace lldb_private;

EmulateInstruction*
EmulateInstruction::FindPlugin (const ArchSpec &arch, const char *plugin_name)
{
    EmulateInstructionCreateInstance create_callback = NULL;
    if (plugin_name)
    {
        create_callback  = PluginManager::GetEmulateInstructionCreateCallbackForPluginName (plugin_name);
        if (create_callback)
        {
           	EmulateInstruction *emulate_insn_ptr = create_callback(arch);
            if (emulate_insn_ptr)
                return emulate_insn_ptr;
        }
    }
    else
    {
        for (uint32_t idx = 0; (create_callback = PluginManager::GetEmulateInstructionCreateCallbackAtIndex(idx)) != NULL; ++idx)
        {
            EmulateInstruction *emulate_insn_ptr = create_callback(arch);
            if (emulate_insn_ptr)
                return emulate_insn_ptr;
        }
    }
    return NULL;
}

EmulateInstruction::EmulateInstruction 
(
    lldb::ByteOrder byte_order,
    uint32_t addr_byte_size,
    const ArchSpec &arch,
    void *baton,
    ReadMemory read_mem_callback,
    WriteMemory write_mem_callback,
    ReadRegister read_reg_callback,
    WriteRegister write_reg_callback
) :
    m_byte_order (endian::InlHostByteOrder()),
    m_addr_byte_size (addr_byte_size),
    m_arch (arch),
    m_baton (baton),
    m_read_mem_callback (read_mem_callback),
    m_write_mem_callback (write_mem_callback),
    m_read_reg_callback (read_reg_callback),
    m_write_reg_callback (write_reg_callback),
    m_opcode (),
    m_opcode_pc (LLDB_INVALID_ADDRESS)
{
}

EmulateInstruction::EmulateInstruction 
(
    lldb::ByteOrder byte_order,
    uint32_t addr_byte_size,
    const ArchSpec &arch
) :
    m_byte_order (endian::InlHostByteOrder()),
    m_addr_byte_size (addr_byte_size),
    m_arch (arch),
    m_baton (NULL),
    m_read_mem_callback (&ReadMemoryDefault),
    m_write_mem_callback (&WriteMemoryDefault),
    m_read_reg_callback (&ReadRegisterDefault),
    m_write_reg_callback (&WriteRegisterDefault),
    m_opcode_pc (LLDB_INVALID_ADDRESS)
{
    ::memset (&m_opcode, 0, sizeof (m_opcode));
}

uint64_t
EmulateInstruction::ReadRegisterUnsigned (uint32_t reg_kind, uint32_t reg_num, uint64_t fail_value, bool *success_ptr)
{
    uint64_t uval64 = 0;
    bool success = m_read_reg_callback (m_baton, reg_kind, reg_num, uval64);
    if (success_ptr)
        *success_ptr = success;
    if (!success)
        uval64 = fail_value;
    return uval64;
}

bool
EmulateInstruction::WriteRegisterUnsigned (const Context &context, uint32_t reg_kind, uint32_t reg_num, uint64_t reg_value)
{
    return m_write_reg_callback (m_baton, context, reg_kind, reg_num, reg_value);    
}

uint64_t
EmulateInstruction::ReadMemoryUnsigned (const Context &context, lldb::addr_t addr, size_t byte_size, uint64_t fail_value, bool *success_ptr)
{
    uint64_t uval64 = 0;
    bool success = false;
    if (byte_size <= 8)
    {
        uint8_t buf[sizeof(uint64_t)];
        size_t bytes_read = m_read_mem_callback (m_baton, context, addr, buf, byte_size);
        if (bytes_read == byte_size)
        {
            uint32_t offset = 0;
            DataExtractor data (buf, byte_size, m_byte_order, m_addr_byte_size);
            uval64 = data.GetMaxU64 (&offset, byte_size);
            success = true;
        }
    }

    if (success_ptr)
        *success_ptr = success;

    if (!success)
        uval64 = fail_value;
    return uval64;
}


bool
EmulateInstruction::WriteMemoryUnsigned (const Context &context, 
                                         lldb::addr_t addr, 
                                         uint64_t uval,
                                         size_t uval_byte_size)
{
    StreamString strm(Stream::eBinary, GetAddressByteSize(), GetByteOrder());
    strm.PutMaxHex64 (uval, uval_byte_size);
    
    size_t bytes_written = m_write_mem_callback (m_baton, context, addr, strm.GetData(), uval_byte_size);
    if (bytes_written == uval_byte_size)
        return true;
    return false;
}


void
EmulateInstruction::SetBaton (void *baton)
{
    m_baton = baton;
}

void
EmulateInstruction::SetCallbacks (ReadMemory read_mem_callback,
                                  WriteMemory write_mem_callback,
                                  ReadRegister read_reg_callback,
                                  WriteRegister write_reg_callback)
{
    m_read_mem_callback = read_mem_callback;
    m_write_mem_callback = write_mem_callback;
    m_read_reg_callback = read_reg_callback;
    m_write_reg_callback = write_reg_callback;
}

void
EmulateInstruction::SetReadMemCallback (ReadMemory read_mem_callback)
{
    m_read_mem_callback = read_mem_callback;
}

                                  
void
EmulateInstruction::SetWriteMemCallback (WriteMemory write_mem_callback)
{
    m_write_mem_callback = write_mem_callback;
}

                                  
void
EmulateInstruction::SetReadRegCallback (ReadRegister read_reg_callback)
{
    m_read_reg_callback = read_reg_callback;
}

                                  
void
EmulateInstruction::SetWriteRegCallback (WriteRegister write_reg_callback)
{
    m_write_reg_callback = write_reg_callback;
}

                                  
                            
//
//  Read & Write Memory and Registers callback functions.
//

size_t 
EmulateInstruction::ReadMemoryFrame (void *baton,
                                     const Context &context, 
                                     lldb::addr_t addr, 
                                     void *dst,
                                     size_t length)
{
    if (!baton)
        return 0;
    
    
    StackFrame *frame = (StackFrame *) baton;

    DataBufferSP data_sp (new DataBufferHeap (length, '\0'));
    Error error;
    
    size_t bytes_read = frame->GetThread().GetProcess().ReadMemory (addr, data_sp->GetBytes(), data_sp->GetByteSize(),
                                                                    error);
    
    if (bytes_read > 0)
        ((DataBufferHeap *) data_sp.get())->CopyData (dst, length);
        
    return bytes_read;
}

size_t 
EmulateInstruction::WriteMemoryFrame (void *baton,
                                      const Context &context, 
                                      lldb::addr_t addr, 
                                      const void *dst,
                                      size_t length)
{
    if (!baton)
        return 0;
    
    StackFrame *frame = (StackFrame *) baton;

    lldb::DataBufferSP data_sp (new DataBufferHeap (dst, length));
    if (data_sp)
    {
        length = data_sp->GetByteSize();
        if (length > 0)
        {
            Error error;
            size_t bytes_written = frame->GetThread().GetProcess().WriteMemory (addr, data_sp->GetBytes(), length, 
                                                                                error);
            
            return bytes_written;
        }
    }
    
    return 0;
}

bool   
EmulateInstruction::ReadRegisterFrame  (void *baton,
                                        uint32_t reg_kind, 
                                        uint32_t reg_num,
                                        uint64_t &reg_value)
{
    if (!baton)
        return false;
        
    StackFrame *frame = (StackFrame *) baton;
    RegisterContext *reg_context = frame->GetRegisterContext().get();
    Scalar value;
    
    uint32_t internal_reg_num = reg_context->ConvertRegisterKindToRegisterNumber (reg_kind, reg_num);
    
    if (internal_reg_num == LLDB_INVALID_REGNUM)
        return false;
    
    if (reg_context->ReadRegisterValue (internal_reg_num, value))
    {
        reg_value = value.GetRawBits64 (0);
        return true;
    }
    
    return false;
}

bool   
EmulateInstruction::WriteRegisterFrame (void *baton,
                                        const Context &context, 
                                        uint32_t reg_kind, 
                                        uint32_t reg_num,
                                        uint64_t reg_value)
{
    if (!baton)
        return false;
        
    StackFrame *frame = (StackFrame *) baton;
    RegisterContext *reg_context = frame->GetRegisterContext().get();
    Scalar value (reg_value);
    
    uint32_t internal_reg_num = reg_context->ConvertRegisterKindToRegisterNumber (reg_kind, reg_num);
    if (internal_reg_num != LLDB_INVALID_REGNUM)
        return reg_context->WriteRegisterValue (internal_reg_num, value);
    else
        return false;
}

size_t 
EmulateInstruction::ReadMemoryDefault (void *baton,
                                       const Context &context, 
                                       lldb::addr_t addr, 
                                       void *dst,
                                       size_t length)
{
    PrintContext ("Read from memory", context);
    fprintf (stdout, "    Read from Memory (address = %p, length = %d)\n",(void *) addr, (uint) length);
    
    *((uint64_t *) dst) = 0xdeadbeef;
    return length;
}

size_t 
EmulateInstruction::WriteMemoryDefault (void *baton,
                                        const Context &context, 
                                        lldb::addr_t addr, 
                                        const void *dst,
                                        size_t length)
{
    PrintContext ("Write to memory", context);
    fprintf (stdout, "    Write to Memory (address = %p, length = %d)\n",  (void *) addr, (uint) length);
    return length;
}

bool   
EmulateInstruction::ReadRegisterDefault  (void *baton,
                                          uint32_t reg_kind, 
                                          uint32_t reg_num,
                                          uint64_t &reg_value)
{
    std::string reg_name;
    TranslateRegister (reg_kind, reg_num, reg_name);
    fprintf (stdout, "  Read Register (%s)\n", reg_name.c_str());
    
    reg_value = 24;
    return true;
}

bool   
EmulateInstruction::WriteRegisterDefault (void *baton,
                                          const Context &context, 
                                          uint32_t reg_kind, 
                                          uint32_t reg_num,
                                          uint64_t reg_value)
{
    PrintContext ("Write to register", context);
    std::string reg_name;
    TranslateRegister (reg_kind, reg_num, reg_name);
    fprintf (stdout, "    Write to Register (%s),  value = 0x%llx\n", reg_name.c_str(), reg_value);
    return true;
}

void
EmulateInstruction::PrintContext (const char *context_type, const Context &context)
{
    switch (context.type)
    {
        case eContextReadOpcode:
            fprintf (stdout, "  %s context: Reading an Opcode\n", context_type);
            break;
            
        case eContextImmediate:
            fprintf (stdout, "  %s context:  Immediate\n", context_type);
            break;
            
        case eContextPushRegisterOnStack:
            fprintf (stdout, "  %s context:  Pushing a register onto the stack.\n", context_type);
            break;
            
        case eContextPopRegisterOffStack:
            fprintf (stdout, "  %s context: Popping a register off the stack.\n", context_type);
            break;
            
        case eContextAdjustStackPointer:
            fprintf (stdout, "  %s context:  Adjusting the stack pointer.\n", context_type);
            break;
            
        case eContextAdjustBaseRegister:
            fprintf (stdout, "  %s context:  Adjusting (writing value back to) a base register.\n", context_type);
            break;
            
        case eContextRegisterPlusOffset:
            fprintf (stdout, "  %s context: Register plus offset\n", context_type);
            break;
            
        case eContextRegisterStore:
            fprintf (stdout, "  %s context:  Storing a register.\n", context_type);
            break;
            
        case eContextRegisterLoad:
            fprintf (stdout, "  %s context:  Loading a register.\n", context_type);
            break;
            
        case eContextRelativeBranchImmediate:
            fprintf (stdout, "  %s context: Relative branch immediate\n", context_type);
            break;
            
        case eContextAbsoluteBranchRegister:
            fprintf (stdout, "  %s context:  Absolute branch register\n", context_type);
            break;
            
        case eContextSupervisorCall:
            fprintf (stdout, "  %s context:  Performing a supervisor call.\n", context_type);
            break;
            
        case eContextTableBranchReadMemory:
            fprintf (stdout, "  %s context:  Table branch read memory\n", context_type);
            break;
            
        case eContextWriteRegisterRandomBits:
            fprintf (stdout, "  %s context:  Write random bits to a register\n", context_type);
            break;
            
        case eContextWriteMemoryRandomBits:
            fprintf (stdout, "  %s context:  Write random bits to a memory address\n", context_type);
            break;
            
        case eContextMultiplication:
            fprintf (stdout, "  %s context:  Performing a multiplication\n", context_type);
            break;
            
        case eContextAddition:
            fprintf (stdout, "  %s context:  Performing an addition\n", context_type);
            break;
            
        case eContextReturnFromException:
            fprintf (stdout, "  %s context:  Returning from an exception\n", context_type);
            break;
            
        default:
            fprintf (stdout, "  %s context:  Unrecognized context.\n", context_type);
            break;
    }
    
    switch (context.info_type)
    {
        case eInfoTypeRegisterPlusOffset:
        {
            std::string reg_name;
            TranslateRegister (context.info.RegisterPlusOffset.reg.kind, 
                               context.info.RegisterPlusOffset.reg.num,
                               reg_name);
            fprintf (stdout, "    Info type:  Register plus offset (%s  +/- %lld)\n", reg_name.c_str(),
                    context.info.RegisterPlusOffset.signed_offset);
        }
            break;
        case eInfoTypeRegisterPlusIndirectOffset:
        {
            std::string base_reg_name;
            std::string offset_reg_name;
            TranslateRegister (context.info.RegisterPlusIndirectOffset.base_reg.kind, 
                                context.info.RegisterPlusIndirectOffset.base_reg.num,
                                base_reg_name);
            TranslateRegister (context.info.RegisterPlusIndirectOffset.offset_reg.kind, 
                                context.info.RegisterPlusIndirectOffset.offset_reg.num,
                                offset_reg_name);
            fprintf (stdout, "    Info type:  Register plus indirect offset (%s  +/- %s)\n", 
                     base_reg_name.c_str(),
                     offset_reg_name.c_str());
        }
            break;
        case eInfoTypeRegisterToRegisterPlusOffset:
        {
            std::string base_reg_name;
            std::string data_reg_name;
            TranslateRegister (context.info.RegisterToRegisterPlusOffset.base_reg.kind, 
                                context.info.RegisterToRegisterPlusOffset.base_reg.num,
                                base_reg_name);
            TranslateRegister (context.info.RegisterToRegisterPlusOffset.data_reg.kind, 
                                context.info.RegisterToRegisterPlusOffset.data_reg.num,
                                data_reg_name);
            fprintf (stdout, "    Info type:  Register plus offset (%s  +/- %lld) and data register (%s)\n", 
                     base_reg_name.c_str(), context.info.RegisterToRegisterPlusOffset.offset,
                     data_reg_name.c_str());
        }
            break;
        case eInfoTypeRegisterToRegisterPlusIndirectOffset:
        {
            std::string base_reg_name;
            std::string offset_reg_name;
            std::string data_reg_name;
            TranslateRegister (context.info.RegisterToRegisterPlusIndirectOffset.base_reg.kind, 
                                context.info.RegisterToRegisterPlusIndirectOffset.base_reg.num,
                                base_reg_name);
            TranslateRegister (context.info.RegisterToRegisterPlusIndirectOffset.offset_reg.kind, 
                                context.info.RegisterToRegisterPlusIndirectOffset.offset_reg.num,
                                offset_reg_name);
            TranslateRegister (context.info.RegisterToRegisterPlusIndirectOffset.data_reg.kind, 
                                context.info.RegisterToRegisterPlusIndirectOffset.data_reg.num,
                                data_reg_name);
            fprintf (stdout, "    Info type:  Register plus indirect offset (%s +/- %s) and data register (%s)\n",
                     base_reg_name.c_str(), offset_reg_name.c_str(), data_reg_name.c_str());
        }
            break;
        
        case eInfoTypeRegisterRegisterOperands:
        {
            std::string op1_reg_name;
            std::string op2_reg_name;
            TranslateRegister (context.info.RegisterRegisterOperands.operand1.kind, 
                                context.info.RegisterRegisterOperands.operand1.num,
                                op1_reg_name);
            TranslateRegister (context.info.RegisterRegisterOperands.operand2.kind, 
                                context.info.RegisterRegisterOperands.operand2.num,
                                op2_reg_name);
            fprintf (stdout, "    Info type:  Register operands for binary op (%s, %s)\n", 
                     op1_reg_name.c_str(),
                     op2_reg_name.c_str());
        }
            break;
        case eInfoTypeOffset:
            fprintf (stdout, "    Info type: signed offset (%lld)\n", context.info.signed_offset);
            break;
            
        case eInfoTypeRegister:
        {
            std::string reg_name;
            TranslateRegister (context.info.reg.kind, context.info.reg.num, reg_name);
            fprintf (stdout, "    Info type:  Register (%s)\n", reg_name.c_str());
        }
            break;
            
        case eInfoTypeImmediate:
            fprintf (stdout, "    Info type:  Immediate (%lld)\n", context.info.immediate);
            break;

        case eInfoTypeImmediateSigned:
            fprintf (stdout, "    Info type:  Signed immediate (%lld)\n", context.info.signed_immediate);
            break;
            
        case eInfoTypeAddress:
            fprintf (stdout, "    Info type:  Address (%p)\n", (void *) context.info.address);
            break;
            
        case eInfoTypeModeAndImmediate:
        {
            std::string mode_name;
            
            if (context.info.ModeAndImmediate.mode == EmulateInstructionARM::eModeARM)
                mode_name = "ARM";
            else if (context.info.ModeAndImmediate.mode == EmulateInstructionARM::eModeThumb)
                mode_name = "Thumb";
            else
                mode_name = "Unknown mode";

            fprintf (stdout, "    Info type:  Mode (%s) and immediate (%d)\n", mode_name.c_str(),
                     context.info.ModeAndImmediate.data_value);
        }
            break;
            
        case eInfoTypeModeAndImmediateSigned:
        {
            std::string mode_name;
            
            if (context.info.ModeAndImmediateSigned.mode == EmulateInstructionARM::eModeARM)
                mode_name = "ARM";
            else if (context.info.ModeAndImmediateSigned.mode == EmulateInstructionARM::eModeThumb)
                mode_name = "Thumb";
            else
                mode_name = "Unknown mode";

            fprintf (stdout, "    Info type:  Mode (%s) and signed immediate (%d)\n", mode_name.c_str(),
                     context.info.ModeAndImmediateSigned.signed_data_value);
        }
            break;
            
        case eInfoTypeMode:
        {
            std::string mode_name;
            
            if (context.info.mode == EmulateInstructionARM::eModeARM)
                mode_name = "ARM";
            else if (context.info.mode == EmulateInstructionARM::eModeThumb)
                mode_name = "Thumb";
            else
                mode_name = "Unknown mode";

            fprintf (stdout, "    Info type:  Mode (%s)\n", mode_name.c_str());
        }
            break;
            
        case eInfoTypeNoArgs:
            fprintf (stdout, "    Info type:  no arguments\n");
            break;

        default:
            break;
    }
}

void
EmulateInstruction::TranslateRegister (uint32_t kind, uint32_t num, std::string &name)
{
    if (kind == eRegisterKindDWARF)
    {
        if (num == 13) //dwarf_sp  NOTE:  This is ARM-SPECIFIC
            name = "sp";
        else if (num == 14) //dwarf_lr  NOTE:  This is ARM-SPECIFIC
            name = "lr";
        else if (num == 15) //dwarf_pc  NOTE:  This is ARM-SPECIFIC
            name = "pc";
        else if (num == 16) //dwarf_cpsr  NOTE:  This is ARM-SPECIFIC
            name = "cpsr";
        else
        {
            StreamString sstr;
            
            sstr.Printf ("r%d", num);
            name = sstr.GetData();
        }
            
    }
    else if (kind == eRegisterKindGeneric)
    {
        if (num == LLDB_REGNUM_GENERIC_SP)
            name = "sp";
        else if (num == LLDB_REGNUM_GENERIC_FLAGS)
            name = "cpsr";
        else if (num == LLDB_REGNUM_GENERIC_PC)
            name = "pc";
        else if (num == LLDB_REGNUM_GENERIC_RA)
            name = "lr";
        else
        {
            StreamString sstr;
            
            sstr.Printf ("r%d", num);
            name = sstr.GetData();
        }
    }
    else
    {
        StreamString sstr;
            
        sstr.Printf ("r%d", num);
        name = sstr.GetData();
    }
}



