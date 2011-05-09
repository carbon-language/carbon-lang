//===-- RegisterContextMacOSXFrameBackchain.cpp -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RegisterContextMacOSXFrameBackchain.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Target/Thread.h"
// Project includes
#include "Utility/StringExtractorGDBRemote.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// RegisterContextMacOSXFrameBackchain constructor
//----------------------------------------------------------------------
RegisterContextMacOSXFrameBackchain::RegisterContextMacOSXFrameBackchain
(
    Thread &thread,
    uint32_t concrete_frame_idx,
    const UnwindMacOSXFrameBackchain::Cursor &cursor
) :
    RegisterContext (thread, concrete_frame_idx),
    m_cursor (cursor),
    m_cursor_is_valid (true)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
RegisterContextMacOSXFrameBackchain::~RegisterContextMacOSXFrameBackchain()
{
}

void
RegisterContextMacOSXFrameBackchain::InvalidateAllRegisters ()
{
    m_cursor_is_valid = false;
}

size_t
RegisterContextMacOSXFrameBackchain::GetRegisterCount ()
{
    return m_thread.GetRegisterContext()->GetRegisterCount();
}

const RegisterInfo *
RegisterContextMacOSXFrameBackchain::GetRegisterInfoAtIndex (uint32_t reg)
{
    return m_thread.GetRegisterContext()->GetRegisterInfoAtIndex(reg);
}

size_t
RegisterContextMacOSXFrameBackchain::GetRegisterSetCount ()
{
    return m_thread.GetRegisterContext()->GetRegisterSetCount();
}



const RegisterSet *
RegisterContextMacOSXFrameBackchain::GetRegisterSet (uint32_t reg_set)
{
    return m_thread.GetRegisterContext()->GetRegisterSet (reg_set);
}



bool
RegisterContextMacOSXFrameBackchain::ReadRegisterValue (uint32_t reg, Scalar &value)
{
    if (!m_cursor_is_valid)
        return false;

    uint64_t reg_value = LLDB_INVALID_ADDRESS;
    
    const RegisterInfo *reg_info = m_thread.GetRegisterContext()->GetRegisterInfoAtIndex (reg);
    if (reg_info == NULL)
        return false;

    switch (reg_info->kinds[eRegisterKindGeneric])
    {
    case LLDB_REGNUM_GENERIC_PC:
        if (m_cursor.pc == LLDB_INVALID_ADDRESS)
            return false;
        reg_value = m_cursor.pc;
        break;
    
    case LLDB_REGNUM_GENERIC_FP:
        if (m_cursor.fp == LLDB_INVALID_ADDRESS)
            return false;
        reg_value = m_cursor.fp;
        break;
    
    default:
        return false;    
    }
    
    switch (reg_info->encoding)
    {
    case eEncodingInvalid:
    case eEncodingVector:
        break;

    case eEncodingUint:
        switch (reg_info->byte_size)
        {
        case 1:
        case 2:
        case 4:
            value = (uint32_t)reg_value;
            return true;

        case 8:
            value = (uint64_t)reg_value;
            return true;
        }
        break;

    case eEncodingSint:
        switch (reg_info->byte_size)
        {
        case 1:
        case 2:
        case 4:
            value = (int32_t)reg_value;
            return true;

        case 8:
            value = (int64_t)reg_value;
            return true;
        }
        break;

    case eEncodingIEEE754:
        switch (reg_info->byte_size)
        {
        case sizeof (float):
            if (sizeof (float) == sizeof(uint32_t))
            {
                value = (uint32_t)reg_value;
                return true;
            }
            else if (sizeof (float) == sizeof(uint64_t))
            {
                value = (uint64_t)reg_value;
                return true;
            }
            break;

        case sizeof (double):
            if (sizeof (double) == sizeof(uint32_t))
            {
                value = (uint32_t)reg_value;
                return true;
            }
            else if (sizeof (double) == sizeof(uint64_t))
            {
                value = (uint64_t)reg_value;
                return true;
            }
            break;

        case sizeof (long double):
            if (sizeof (long double) == sizeof(uint32_t))
            {
                value = (uint32_t)reg_value;
                return true;
            }
            else if (sizeof (long double) == sizeof(uint64_t))
            {
                value = (uint64_t)reg_value;
                return true;
            }
            break;
        }
        break;
    }
    return false;
}


bool
RegisterContextMacOSXFrameBackchain::ReadRegisterBytes (uint32_t reg, DataExtractor &data)
{
    Scalar reg_value;
    
    if (ReadRegisterValue (reg, reg_value))
    {
        if (reg_value.GetData(data))
        {
            // "reg_value" is local and now "data" points to the data within
            // "reg_value", so we must make a copy that will live within "data"
            DataBufferSP data_sp (new DataBufferHeap (data.GetDataStart(), data.GetByteSize()));
            data.SetData (data_sp, 0, data.GetByteSize());
            return true;
        }
    }
    return false;
}


bool
RegisterContextMacOSXFrameBackchain::WriteRegisterValue (uint32_t reg, const Scalar &value)
{
    // Not supported yet. We could easily add support for this by remembering
    // the address of each entry (it would need to be part of the cursor)
    return false;
}


bool
RegisterContextMacOSXFrameBackchain::WriteRegisterBytes (uint32_t reg, DataExtractor &data, uint32_t data_offset)
{
    // Not supported yet. We could easily add support for this by remembering
    // the address of each entry (it would need to be part of the cursor)
    return false;
}


bool
RegisterContextMacOSXFrameBackchain::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    // libunwind frames can't handle this it doesn't always have all register
    // values. This call should only be called on frame zero anyway so there
    // shouldn't be any problem
    return false;
}

bool
RegisterContextMacOSXFrameBackchain::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    // Since this class doesn't respond to "ReadAllRegisterValues()", it must
    // not have been the one that saved all the register values. So we just let
    // the thread's register context (the register context for frame zero) do
    // the writing.
    return m_thread.GetRegisterContext()->WriteAllRegisterValues(data_sp);
}


uint32_t
RegisterContextMacOSXFrameBackchain::ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num)
{
    return m_thread.GetRegisterContext()->ConvertRegisterKindToRegisterNumber (kind, num);
}

