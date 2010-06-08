//===-- LibUnwindRegisterContext.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LibUnwindRegisterContext.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Target/Thread.h"
// Project includes

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// LibUnwindRegisterContext constructor
//----------------------------------------------------------------------
LibUnwindRegisterContext::LibUnwindRegisterContext
(
    Thread &thread,
    StackFrame *frame,
    const lldb_private::unw_cursor_t& unwind_cursor
) :
    RegisterContext (thread, frame),
    m_unwind_cursor (unwind_cursor),
    m_unwind_cursor_is_valid (true)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
LibUnwindRegisterContext::~LibUnwindRegisterContext()
{
}

void
LibUnwindRegisterContext::Invalidate ()
{
    m_unwind_cursor_is_valid = false;
}

size_t
LibUnwindRegisterContext::GetRegisterCount ()
{
    return m_thread.GetRegisterContext()->GetRegisterCount();
}

const lldb::RegisterInfo *
LibUnwindRegisterContext::GetRegisterInfoAtIndex (uint32_t reg)
{
    return m_thread.GetRegisterContext()->GetRegisterInfoAtIndex(reg);
}

size_t
LibUnwindRegisterContext::GetRegisterSetCount ()
{
    return m_thread.GetRegisterContext()->GetRegisterSetCount();
}



const lldb::RegisterSet *
LibUnwindRegisterContext::GetRegisterSet (uint32_t reg_set)
{
    return m_thread.GetRegisterContext()->GetRegisterSet (reg_set);
}



bool
LibUnwindRegisterContext::ReadRegisterValue (uint32_t reg, Scalar &value)
{
    if (m_unwind_cursor_is_valid == false)
        return false;

    // Read the register
    unw_word_t reg_value;
    if (unw_get_reg (&m_unwind_cursor, reg, &reg_value) != UNW_ESUCCESS)
        return false;

    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
    switch (reg_info->encoding)
    {
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
        if (reg_info->byte_size > sizeof(unw_word_t))
            return false;

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
LibUnwindRegisterContext::ReadRegisterBytes (uint32_t reg, DataExtractor &data)
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
LibUnwindRegisterContext::WriteRegisterValue (uint32_t reg, const Scalar &value)
{
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
    if (reg_info == NULL)
        return false;
    unw_word_t reg_value;
    switch (value.GetType())
    {
    case Scalar::e_sint:        reg_value = value.SInt(); break;
    case Scalar::e_uint:        reg_value = value.UInt(); break;
    case Scalar::e_slong:       reg_value = value.SLong(); break;
    case Scalar::e_ulong:       reg_value = value.ULong(); break;
    case Scalar::e_slonglong:   reg_value = value.SLongLong(); break;
    case Scalar::e_ulonglong:   reg_value = value.ULongLong(); break;
    case Scalar::e_float:
        if (sizeof (float) == sizeof (unsigned int))
            reg_value = value.UInt();
        else if (sizeof (float) == sizeof (unsigned long))
            reg_value = value.ULong();
        else if (sizeof (float) == sizeof (unsigned long long))
            reg_value = value.ULongLong();
        else
            return false;
        break;
            
    case Scalar::e_double:
        if (sizeof (double) == sizeof (unsigned int))
            reg_value = value.UInt();
        else if (sizeof (double) == sizeof (unsigned long))
            reg_value = value.ULong();
        else if (sizeof (double) == sizeof (unsigned long long))
            reg_value = value.ULongLong();
        else
            return false;
        break;
            
    case Scalar::e_long_double:
        if (sizeof (long double) == sizeof (unsigned int))
            reg_value = value.UInt();
        else if (sizeof (long double) == sizeof (unsigned long))
            reg_value = value.ULong();
        else if (sizeof (long double) == sizeof (unsigned long long))
            reg_value = value.ULongLong();
        else
            return false;
        break;
    }
    
    return unw_set_reg (&m_unwind_cursor, reg, reg_value) == UNW_ESUCCESS;
}


bool
LibUnwindRegisterContext::WriteRegisterBytes (uint32_t reg, DataExtractor &data, uint32_t data_offset)
{
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);

    if (reg_info == NULL)
        return false;
    if (reg_info->byte_size > sizeof (unw_word_t))
        return false;
    
    Scalar value;
    uint32_t offset = data_offset;
    
    switch (reg_info->encoding)
    {
    case eEncodingUint:
        if (reg_info->byte_size <= 4)
            value = data.GetMaxU32 (&offset, reg_info->byte_size);
        else if (reg_info->byte_size <= 8)
            value = data.GetMaxU64 (&offset, reg_info->byte_size);
        else
            return false;
        break;

    case eEncodingSint:
        if (reg_info->byte_size <= 4)
            value = (int32_t)data.GetMaxU32 (&offset, reg_info->byte_size);
        else if (reg_info->byte_size <= 8)
            value = data.GetMaxS64 (&offset, reg_info->byte_size);
        else
            return false;
        break;

    case eEncodingIEEE754:
        switch (reg_info->byte_size)
        {
        case sizeof (float):
            value = data.GetFloat (&offset);
            break;

        case sizeof (double):
            value = data.GetDouble (&offset);
            break;

        case sizeof (long double):
            value = data.GetLongDouble (&offset);
            break;
        default:
            return false;
        }
    }        
    return WriteRegisterValue (reg, value);
}


bool
LibUnwindRegisterContext::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    // libunwind frames can't handle this it doesn't always have all register
    // values. This call should only be called on frame zero anyway so there
    // shouldn't be any problem
    return false;
}

bool
LibUnwindRegisterContext::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    // Since this class doesn't respond to "ReadAllRegisterValues()", it must
    // not have been the one that saved all the register values. So we just let
    // the thread's register context (the register context for frame zero) do
    // the writing.
    return m_thread.GetRegisterContext()->WriteAllRegisterValues(data_sp);
}


uint32_t
LibUnwindRegisterContext::ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num)
{
    return m_thread.GetRegisterContext()->ConvertRegisterKindToRegisterNumber (kind, num);
}

