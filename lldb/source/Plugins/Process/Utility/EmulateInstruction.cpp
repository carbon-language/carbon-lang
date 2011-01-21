//===-- EmulateInstruction.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "EmulateInstruction.h"

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/StreamString.h"
using namespace lldb;
using namespace lldb_private;


EmulateInstruction::EmulateInstruction 
(
    lldb::ByteOrder byte_order,
    uint32_t addr_byte_size,
    void *baton,
    ReadMemory read_mem_callback,
    WriteMemory write_mem_callback,
    ReadRegister read_reg_callback,
    WriteRegister write_reg_callback
) :
    m_byte_order (lldb::eByteOrderHost),
    m_addr_byte_size (sizeof (void *)),
    m_baton (baton),
    m_read_mem_callback (read_mem_callback),
    m_write_mem_callback (write_mem_callback),
    m_read_reg_callback (read_reg_callback),
    m_write_reg_callback (write_reg_callback),
    m_inst_pc (LLDB_INVALID_ADDRESS)
{
    ::bzero (&m_inst, sizeof (m_inst));
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
