//===-- RegisterValue.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/RegisterValue.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Stream.h"
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;


bool
RegisterValue::Dump (Stream *s, const RegisterInfo *reg_info, bool prefix_with_name, Format format) const
{
    DataExtractor data;
    if (GetData (data))
    {
        if (prefix_with_name && reg_info->name != NULL)
            s->Printf ("%s = ", reg_info->name);
        if (format == eFormatDefault)
            format = reg_info->format;

        data.Dump (s, 
                   0,                       // Offset in "data"
                   format,                  // Format to use when dumping
                   reg_info->byte_size,        // item_byte_size
                   1,                       // item_count
                   UINT32_MAX,              // num_per_line
                   LLDB_INVALID_ADDRESS,    // base_addr
                   0,                       // item_bit_size
                   0);                      // item_bit_offset
        return true;
    }
    return false;
}


bool
RegisterValue::GetData (DataExtractor &data) const
{
    return data.SetData(GetBytes(), GetByteSize(), GetByteOrder()) > 0;
}


uint32_t
RegisterValue::GetAsMemoryData (const RegisterInfo *reg_info,
                                void *dst,
                                uint32_t dst_len, 
                                lldb::ByteOrder dst_byte_order,
                                Error &error) const
{    
    if (reg_info == NULL)
    {
        error.SetErrorString ("invalid register info argument.");
        return 0;
    }
    
    // ReadRegister should have already been called on tgus object prior to 
    // calling this.
    if (GetType() == eTypeInvalid)
    {
        // No value has been read into this object...
        error.SetErrorStringWithFormat("invalid register value type for register %s", reg_info->name);
        return 0;
    }
    
    if (dst_len > kMaxRegisterByteSize)
    {
        error.SetErrorString ("destination is too big");
        return 0;
    }
    
    const uint32_t src_len = reg_info->byte_size;
    
    // Extract the register data into a data extractor
    DataExtractor reg_data;
    if (!GetData(reg_data))
    {
        error.SetErrorString ("invalid register value to copy into");
        return 0;
    }
    
    // Prepare a memory buffer that contains some or all of the register value
    const uint32_t bytes_copied = reg_data.CopyByteOrderedData (0,                  // src offset
                                                                src_len,            // src length
                                                                dst,                // dst buffer
                                                                dst_len,            // dst length
                                                                dst_byte_order);    // dst byte order
    if (bytes_copied == 0) 
        error.SetErrorStringWithFormat("failed to copy data for register write of %s", reg_info->name);
    
    return bytes_copied;
}

uint32_t
RegisterValue::SetFromMemoryData (const RegisterInfo *reg_info,
                                  const void *src,
                                  uint32_t src_len,
                                  lldb::ByteOrder src_byte_order,
                                  Error &error)
{
    if (reg_info == NULL)
    {
        error.SetErrorString ("invalid register info argument.");
        return 0;
    }
    
    // Moving from addr into a register
    //
    // Case 1: src_len == dst_len
    //
    //   |AABBCCDD| Address contents
    //   |AABBCCDD| Register contents
    //
    // Case 2: src_len > dst_len
    //
    //   Error!  (The register should always be big enough to hold the data)
    //
    // Case 3: src_len < dst_len
    //
    //   |AABB| Address contents
    //   |AABB0000| Register contents [on little-endian hardware]
    //   |0000AABB| Register contents [on big-endian hardware]
    if (src_len > kMaxRegisterByteSize)
    {
        error.SetErrorStringWithFormat ("register buffer is too small to receive %u bytes of data.", src_len);
        return 0;
    }
    
    const uint32_t dst_len = reg_info->byte_size;
    
    if (src_len > dst_len)
    {
        error.SetErrorStringWithFormat("%u bytes is too big to store in register %s (%u bytes)", src_len, reg_info->name, dst_len);
        return 0;
    }

    // Use a data extractor to correctly copy and pad the bytes read into the
    // register value
    DataExtractor src_data (src, src_len, src_byte_order, 4);
    
    // Given the register info, set the value type of this RegisterValue object
    SetType (reg_info);
    // And make sure we were able to figure out what that register value was
    RegisterValue::Type value_type = GetType();
    if (value_type == eTypeInvalid)        
    {
        // No value has been read into this object...
        error.SetErrorStringWithFormat("invalid register value type for register %s", reg_info->name);
        return 0;
    }
    else if (value_type == eTypeBytes)
    {
        m_data.buffer.byte_order = src_byte_order;
    }

    const uint32_t bytes_copied = src_data.CopyByteOrderedData (0,               // src offset
                                                                src_len,         // src length
                                                                GetBytes(),      // dst buffer
                                                                GetByteSize(),   // dst length
                                                                GetByteOrder()); // dst byte order
    if (bytes_copied == 0)
        error.SetErrorStringWithFormat("failed to copy data for register write of %s", reg_info->name);

    return bytes_copied;
}

bool
RegisterValue::GetScalarValue (Scalar &scalar) const
{
    switch (m_type)
    {
        case eTypeInvalid:      break;
        case eTypeBytes:        break;
        case eTypeUInt8:        scalar = m_data.uint8; return true;
        case eTypeUInt16:       scalar = m_data.uint16; return true;
        case eTypeUInt32:       scalar = m_data.uint32; return true;
        case eTypeUInt64:       scalar = m_data.uint64; return true;
#if defined (ENABLE_128_BIT_SUPPORT)
        case eTypeUInt128:      break;
#endif
        case eTypeFloat:        scalar = m_data.ieee_float; return true;
        case eTypeDouble:       scalar = m_data.ieee_double; return true;
        case eTypeLongDouble:   scalar = m_data.ieee_long_double; return true;
    }
    return false;
}

void
RegisterValue::Clear()
{
    m_type = eTypeInvalid;
}

RegisterValue::Type
RegisterValue::SetType (const RegisterInfo *reg_info)
{
    m_type = eTypeInvalid;
    const uint32_t byte_size = reg_info->byte_size;
    switch (reg_info->encoding)
    {
        case eEncodingInvalid:
            break;
            
        case eEncodingUint:
        case eEncodingSint:
            if (byte_size == 1)
                m_type = eTypeUInt8;
            else if (byte_size <= 2)
                m_type = eTypeUInt16;
            else if (byte_size <= 4)
                m_type = eTypeUInt32;
            else if (byte_size <= 8)
                m_type = eTypeUInt64;
#if defined (ENABLE_128_BIT_SUPPORT)
            else if (byte_size <= 16)
                m_type = eTypeUInt128;
#endif
            break;

        case eEncodingIEEE754:
            if (byte_size == sizeof(float))
                m_type = eTypeFloat;
            if (byte_size == sizeof(double))
                m_type = eTypeDouble;
            if (byte_size == sizeof(long double))
                m_type = eTypeLongDouble;
            break;

        case eEncodingVector:
            m_type = eTypeBytes;
            break;
    }
    return m_type;
}

Error
RegisterValue::SetValueFromData (const RegisterInfo *reg_info, DataExtractor &src, uint32_t src_offset, bool partial_data_ok)
{
    Error error;
    
    if (src.GetByteSize() == 0)
    {
        error.SetErrorString ("empty data.");
        return error;
    }

    if (reg_info->byte_size == 0)
    {
        error.SetErrorString ("invalid register info.");
        return error;
    }

    uint32_t src_len = src.GetByteSize() - src_offset;
    
    if (!partial_data_ok && (src_len < reg_info->byte_size))
    {
        error.SetErrorString ("not enough data.");
        return error;
    }
        
    // Cap the data length if there is more than enough bytes for this register
    // value
    if (src_len > reg_info->byte_size)
        src_len = reg_info->byte_size;

    // Zero out the value in case we get partial data...
    memset (m_data.buffer.bytes, 0, sizeof (m_data.buffer.bytes));
    
    switch (SetType (reg_info))
    {
        case eTypeInvalid:
            error.SetErrorString("");
            break;
        case eTypeUInt8:    SetUInt8  (src.GetMaxU32 (&src_offset, src_len)); break;
        case eTypeUInt16:   SetUInt16 (src.GetMaxU32 (&src_offset, src_len)); break;
        case eTypeUInt32:   SetUInt32 (src.GetMaxU32 (&src_offset, src_len)); break;
        case eTypeUInt64:   SetUInt64 (src.GetMaxU64 (&src_offset, src_len)); break;
#if defined (ENABLE_128_BIT_SUPPORT)
        case eTypeUInt128:
            {
                __uint128_t data1 = src.GetU64 (&src_offset);
                __uint128_t data2 = src.GetU64 (&src_offset);
                if (src.GetByteSize() == eByteOrderBig)
                    SetUInt128 (data1 << 64 + data2);
                else
                    SetUInt128 (data2 << 64 + data1);
            }
            break;
#endif
        case eTypeFloat:        SetFloat (src.GetFloat (&src_offset));      break;
        case eTypeDouble:       SetDouble(src.GetDouble (&src_offset));     break;
        case eTypeLongDouble:   SetFloat (src.GetLongDouble (&src_offset)); break;
        case eTypeBytes:
        {
            m_data.buffer.length = reg_info->byte_size;
            m_data.buffer.byte_order = src.GetByteOrder();
            assert (m_data.buffer.length <= kMaxRegisterByteSize);
            if (m_data.buffer.length > kMaxRegisterByteSize)
                m_data.buffer.length = kMaxRegisterByteSize;
            if (src.CopyByteOrderedData (src_offset,                    // offset within "src" to start extracting data
                                         src_len,                       // src length
                                         m_data.buffer.bytes,           // dst buffer
                                         m_data.buffer.length,          // dst length
                                         m_data.buffer.byte_order) == 0)// dst byte order
            {
                error.SetErrorString ("data copy failed data.");
                return error;
            }
        }
    }
    
    return error;
}

Error
RegisterValue::SetValueFromCString (const RegisterInfo *reg_info, const char *value_str)
{
    Error error;
    if (reg_info == NULL)
    {
        error.SetErrorString ("Invalid register info argument.");
        return error;
    }

    if (value_str == NULL && value_str[0] == '\0')
    {
        error.SetErrorString ("Invalid c-string value string.");
        return error;
    }
    bool success = false;
    const uint32_t byte_size = reg_info->byte_size;
    switch (reg_info->encoding)
    {
        default:
        case eEncodingInvalid:
            error.SetErrorString ("Invalid encoding.");
            break;
            
        case eEncodingUint:
            if (byte_size <= sizeof (uint64_t))
            {
                uint64_t uval64 = Args::StringToUInt64(value_str, UINT64_MAX, 0, &success);
                if (!success)
                    error.SetErrorStringWithFormat ("'%s' is not a valid unsigned integer string value.\n", value_str);
                else if (!Args::UInt64ValueIsValidForByteSize (uval64, byte_size))
                    error.SetErrorStringWithFormat ("Value 0x%llx is too large to fit in a %u byte unsigned integer value.\n", uval64, byte_size);
                else
                {
                    if (!SetUInt (uval64, reg_info->byte_size))
                        error.SetErrorStringWithFormat ("Unsupported unsigned integer byte size: %u.\n", byte_size);
                }
            }
            else
            {
                error.SetErrorStringWithFormat ("Unsupported unsigned integer byte size: %u.\n", byte_size);
                return error;
            }
            break;
            
        case eEncodingSint:
            if (byte_size <= sizeof (long long))
            {
                uint64_t sval64 = Args::StringToSInt64(value_str, INT64_MAX, 0, &success);
                if (!success)
                    error.SetErrorStringWithFormat ("'%s' is not a valid signed integer string value.\n", value_str);
                else if (!Args::SInt64ValueIsValidForByteSize (sval64, byte_size))
                    error.SetErrorStringWithFormat ("Value 0x%llx is too large to fit in a %u byte signed integer value.\n", sval64, byte_size);
                else
                {
                    if (!SetUInt (sval64, reg_info->byte_size))
                        error.SetErrorStringWithFormat ("Unsupported signed integer byte size: %u.\n", byte_size);
                }
            }
            else
            {
                error.SetErrorStringWithFormat ("Unsupported signed integer byte size: %u.\n", byte_size);
                return error;
            }
            break;
            
        case eEncodingIEEE754:
            if (byte_size == sizeof (float))
            {
                if (::sscanf (value_str, "%f", &m_data.ieee_float) == 1)
                    m_type = eTypeFloat;
                else
                    error.SetErrorStringWithFormat ("'%s' is not a valid float string value.\n", value_str);
            }
            else if (byte_size == sizeof (double))
            {
                if (::sscanf (value_str, "%lf", &m_data.ieee_double) == 1)
                    m_type = eTypeDouble;
                else
                    error.SetErrorStringWithFormat ("'%s' is not a valid float string value.\n", value_str);
            }
            else if (byte_size == sizeof (long double))
            {
                if (::sscanf (value_str, "%Lf", &m_data.ieee_long_double) == 1)
                    m_type = eTypeLongDouble;
                else
                    error.SetErrorStringWithFormat ("'%s' is not a valid float string value.\n", value_str);
            }
            else
            {
                error.SetErrorStringWithFormat ("Unsupported float byte size: %u.\n", byte_size);
                return error;
            }
            break;
            
        case eEncodingVector:
            error.SetErrorString ("Vector encoding unsupported.");
            break;
    }
    if (error.Fail())
        m_type = eTypeInvalid;
    
    return error;
}


bool
RegisterValue::SignExtend (uint32_t sign_bitpos)
{
    switch (m_type)
    {
        case eTypeInvalid:
            break;

        case eTypeUInt8:        
            if (sign_bitpos == (8-1))
                return true;
            else if (sign_bitpos < (8-1))
            {
                uint8_t sign_bit = 1u << sign_bitpos;
                if (m_data.uint8 & sign_bit)
                {
                    const uint8_t mask = ~(sign_bit) + 1u;
                    m_data.uint8 |= mask;
                }
                return true;
            }
            break;

        case eTypeUInt16:
            if (sign_bitpos == (16-1))
                return true;
            else if (sign_bitpos < (16-1))
            {
                uint16_t sign_bit = 1u << sign_bitpos;
                if (m_data.uint16 & sign_bit)
                {
                    const uint16_t mask = ~(sign_bit) + 1u;
                    m_data.uint16 |= mask;
                }
                return true;
            }
            break;
        
        case eTypeUInt32:
            if (sign_bitpos == (32-1))
                return true;
            else if (sign_bitpos < (32-1))
            {
                uint32_t sign_bit = 1u << sign_bitpos;
                if (m_data.uint32 & sign_bit)
                {
                    const uint32_t mask = ~(sign_bit) + 1u;
                    m_data.uint32 |= mask;
                }
                return true;
            }
            break;

        case eTypeUInt64:
            if (sign_bitpos == (64-1))
                return true;
            else if (sign_bitpos < (64-1))
            {
                uint64_t sign_bit = 1ull << sign_bitpos;
                if (m_data.uint64 & sign_bit)
                {
                    const uint64_t mask = ~(sign_bit) + 1ull;
                    m_data.uint64 |= mask;
                }
                return true;
            }
            break;

#if defined (ENABLE_128_BIT_SUPPORT)
        case eTypeUInt128:
            if (sign_bitpos == (128-1))
                return true;
            else if (sign_bitpos < (128-1))
            {
                __uint128_t sign_bit = (__uint128_t)1u << sign_bitpos;
                if (m_data.uint128 & sign_bit)
                {
                    const uint128_t mask = ~(sign_bit) + 1u;
                    m_data.uint128 |= mask;
                }
                return true;
            }
            break;
#endif
        case eTypeFloat:
        case eTypeDouble:
        case eTypeLongDouble:
        case eTypeBytes:
            break;
    }
    return false;
}

bool
RegisterValue::CopyValue (const RegisterValue &rhs)
{
    m_type = rhs.m_type;
    switch (m_type)
    {
        default:
        case eTypeInvalid: 
            return false;
        case eTypeUInt8:        m_data.uint8 = rhs.m_data.uint8; break;
        case eTypeUInt16:       m_data.uint16 = rhs.m_data.uint16; break;
        case eTypeUInt32:       m_data.uint32 = rhs.m_data.uint32; break;
        case eTypeUInt64:       m_data.uint64 = rhs.m_data.uint64; break;
#if defined (ENABLE_128_BIT_SUPPORT)
        case eTypeUInt128:      m_data.uint128 = rhs.m_data.uint128; break;
#endif
        case eTypeFloat:        m_data.ieee_float = rhs.m_data.ieee_float; break;
        case eTypeDouble:       m_data.ieee_double = rhs.m_data.ieee_double; break;
        case eTypeLongDouble:   m_data.ieee_long_double = rhs.m_data.ieee_long_double; break;
        case eTypeBytes:        
            assert (rhs.m_data.buffer.length <= kMaxRegisterByteSize);
            ::memcpy (m_data.buffer.bytes, rhs.m_data.buffer.bytes, kMaxRegisterByteSize);
            m_data.buffer.length = rhs.m_data.buffer.length;
            m_data.buffer.byte_order = rhs.m_data.buffer.byte_order;
            break;
    }
    return true;
}

uint16_t
RegisterValue::GetAsUInt16 (uint16_t fail_value, bool *success_ptr) const
{
    if (success_ptr)
        *success_ptr = true;
    
    switch (m_type)
    {
        default:            break;
        case eTypeUInt8:    return m_data.uint8;
        case eTypeUInt16:   return m_data.uint16;
    }
    if (success_ptr)
        *success_ptr = false;
    return fail_value;
}

uint32_t
RegisterValue::GetAsUInt32 (uint32_t fail_value, bool *success_ptr) const
{
    if (success_ptr)
        *success_ptr = true;
    switch (m_type)
    {
        default:            break;
        case eTypeUInt8:    return m_data.uint8;
        case eTypeUInt16:   return m_data.uint16;
        case eTypeUInt32:   return m_data.uint32;
        case eTypeFloat:
            if (sizeof(float) == sizeof(uint32_t))
                return m_data.uint32;
            break;
        case eTypeDouble:
            if (sizeof(double) == sizeof(uint32_t))
                return m_data.uint32;
            break;
        case eTypeLongDouble:
            if (sizeof(long double) == sizeof(uint32_t))
                return m_data.uint32;
            break;
    }
    if (success_ptr)
        *success_ptr = false;
    return fail_value;
}

uint64_t
RegisterValue::GetAsUInt64 (uint64_t fail_value, bool *success_ptr) const
{
    if (success_ptr)
        *success_ptr = true;
    switch (m_type)
    {
        default:            break;
        case eTypeUInt8:    return m_data.uint8;
        case eTypeUInt16:   return m_data.uint16;
        case eTypeUInt32:   return m_data.uint32;
        case eTypeUInt64:   return m_data.uint64;
        case eTypeFloat:
            if (sizeof(float) == sizeof(uint64_t))
                return m_data.uint64;
            break;
        case eTypeDouble:
            if (sizeof(double) == sizeof(uint64_t))
                return m_data.uint64;
            break;
        case eTypeLongDouble:
            if (sizeof(long double) == sizeof(uint64_t))
                return m_data.uint64;
            break;
    }
    if (success_ptr)
        *success_ptr = false;
    return fail_value;
}

#if defined (ENABLE_128_BIT_SUPPORT)
__uint128_t
RegisterValue::GetAsUInt128 (__uint128_t fail_value, bool *success_ptr) const
{
    if (success_ptr)
        *success_ptr = true;
    switch (m_type)
    {
        default:            break;
        case eTypeUInt8:    return m_data.uint8;
        case eTypeUInt16:   return m_data.uint16;
        case eTypeUInt32:   return m_data.uint32;
        case eTypeUInt64:   return m_data.uint64;
        case eTypeUInt128:  return m_data.uint128;
        case eTypeFloat:
            if (sizeof(float) == sizeof(__uint128_t))
                return m_data.uint128;
            break;
        case eTypeDouble:
            if (sizeof(double) == sizeof(__uint128_t))
                return m_data.uint128;
            break;
        case eTypeLongDouble:
            if (sizeof(long double) == sizeof(__uint128_t))
                return m_data.uint128;
            break;
    }
    if (success_ptr)
        *success_ptr = false;
    return fail_value;
}
#endif
float
RegisterValue::GetAsFloat (float fail_value, bool *success_ptr) const
{
    if (success_ptr)
        *success_ptr = true;
    switch (m_type)
    {
        default:            break;
        case eTypeUInt32:
            if (sizeof(float) == sizeof(m_data.uint32))
                return m_data.ieee_float;
            break;
        case eTypeUInt64:
            if (sizeof(float) == sizeof(m_data.uint64))
                return m_data.ieee_float;
            break;
#if defined (ENABLE_128_BIT_SUPPORT)
        case eTypeUInt128:
            if (sizeof(float) == sizeof(m_data.uint128))
                return m_data.ieee_float;
            break;
#endif
        case eTypeFloat:    return m_data.ieee_float;
        case eTypeDouble:
            if (sizeof(float) == sizeof(double))
                return m_data.ieee_float;
            break;
        case eTypeLongDouble:
            if (sizeof(float) == sizeof(long double))
                return m_data.ieee_float;
            break;
    }
    if (success_ptr)
        *success_ptr = false;
    return fail_value;
}

double
RegisterValue::GetAsDouble (double fail_value, bool *success_ptr) const
{
    if (success_ptr)
        *success_ptr = true;
    switch (m_type)
    {
        default:            
            break;
            
        case eTypeUInt32:
            if (sizeof(double) == sizeof(m_data.uint32))
                return m_data.ieee_double;
            break;
            
        case eTypeUInt64:
            if (sizeof(double) == sizeof(m_data.uint64))
                return m_data.ieee_double;
            break;
            
#if defined (ENABLE_128_BIT_SUPPORT)
        case eTypeUInt128:
            if (sizeof(double) == sizeof(m_data.uint128))
                return m_data.ieee_double;
#endif
        case eTypeFloat:    return m_data.ieee_float;
        case eTypeDouble:   return m_data.ieee_double;
            
        case eTypeLongDouble:
            if (sizeof(double) == sizeof(long double))
                return m_data.ieee_double;
            break;
    }
    if (success_ptr)
        *success_ptr = false;
    return fail_value;
}

long double
RegisterValue::GetAsLongDouble (long double fail_value, bool *success_ptr) const
{
    if (success_ptr)
        *success_ptr = true;
    switch (m_type)
    {
        default:
            break;
            
        case eTypeUInt32:
            if (sizeof(long double) == sizeof(m_data.uint32))
                return m_data.ieee_long_double;
            break;
            
        case eTypeUInt64:
            if (sizeof(long double) == sizeof(m_data.uint64))
                return m_data.ieee_long_double;
            break;
            
#if defined (ENABLE_128_BIT_SUPPORT)
        case eTypeUInt128:
            if (sizeof(long double) == sizeof(m_data.uint128))
                return m_data.ieee_long_double;
#endif
        case eTypeFloat:        return m_data.ieee_float;
        case eTypeDouble:       return m_data.ieee_double;
        case eTypeLongDouble:   return m_data.ieee_long_double;
            break;
    }
    if (success_ptr)
        *success_ptr = false;
    return fail_value;
}

const void *
RegisterValue::GetBytes () const
{
    switch (m_type)
    {
        case eTypeInvalid:      break;
        case eTypeUInt8:        return &m_data.uint8;
        case eTypeUInt16:       return &m_data.uint16;
        case eTypeUInt32:       return &m_data.uint32;
        case eTypeUInt64:       return &m_data.uint64;
#if defined (ENABLE_128_BIT_SUPPORT)
        case eTypeUInt128:      return &m_data.uint128;
#endif
        case eTypeFloat:        return &m_data.ieee_float;
        case eTypeDouble:       return &m_data.ieee_double;
        case eTypeLongDouble:   return &m_data.ieee_long_double;
        case eTypeBytes:        return m_data.buffer.bytes;
    }
    return NULL;
}

void *
RegisterValue::GetBytes ()
{
    switch (m_type)
    {
        case eTypeInvalid:      break;
        case eTypeUInt8:        return &m_data.uint8;
        case eTypeUInt16:       return &m_data.uint16;
        case eTypeUInt32:       return &m_data.uint32;
        case eTypeUInt64:       return &m_data.uint64;
#if defined (ENABLE_128_BIT_SUPPORT)
        case eTypeUInt128:      return &m_data.uint128;
#endif
        case eTypeFloat:        return &m_data.ieee_float;
        case eTypeDouble:       return &m_data.ieee_double;
        case eTypeLongDouble:   return &m_data.ieee_long_double;
        case eTypeBytes:        return m_data.buffer.bytes;
    }
    return NULL;
}

uint32_t
RegisterValue::GetByteSize () const
{
    switch (m_type)
    {
        case eTypeInvalid: break;
        case eTypeUInt8:        return sizeof(m_data.uint8);
        case eTypeUInt16:       return sizeof(m_data.uint16);
        case eTypeUInt32:       return sizeof(m_data.uint32);
        case eTypeUInt64:       return sizeof(m_data.uint64);
#if defined (ENABLE_128_BIT_SUPPORT)
        case eTypeUInt128:      return sizeof(m_data.uint128);
#endif
        case eTypeFloat:        return sizeof(m_data.ieee_float);
        case eTypeDouble:       return sizeof(m_data.ieee_double);
        case eTypeLongDouble:   return sizeof(m_data.ieee_long_double);
        case eTypeBytes: return m_data.buffer.length;
    }
    return 0;
}


bool
RegisterValue::SetUInt (uint64_t uint, uint32_t byte_size)
{
    if (byte_size == 0)
    {
        SetUInt64 (uint);
    }
    else if (byte_size == 1)
    {
        SetUInt8 (uint);
    }
    else if (byte_size <= 2)
    {
        SetUInt16 (uint);
    }
    else if (byte_size <= 4)
    {
        SetUInt32 (uint);
    }
    else if (byte_size <= 8)
    {
        SetUInt64 (uint);
    }
#if defined (ENABLE_128_BIT_SUPPORT)
    else if (byte_size <= 16)
    {
        SetUInt128 (uint);
    }
#endif
    else
        return false;
    return true;
}

void
RegisterValue::SetBytes (const void *bytes, size_t length, lldb::ByteOrder byte_order)
{
    // If this assertion fires off we need to increase the size of
    // m_data.buffer.bytes, or make it something that is allocated on
    // the heap. Since the data buffer is in a union, we can't make it
    // a collection class like SmallVector...
    assert (length <= sizeof (m_data.buffer.bytes));
    if (bytes && length > 0)
    {
        m_type = eTypeBytes;
        m_data.buffer.length = length;
        assert (length < sizeof (m_data.buffer.bytes));
        memcpy (m_data.buffer.bytes, bytes, length);
        m_data.buffer.byte_order = byte_order;
    }
    else
    {
        m_type = eTypeInvalid;
        m_data.buffer.length = 0;
    }
}

