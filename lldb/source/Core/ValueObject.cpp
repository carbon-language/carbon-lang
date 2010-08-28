//===-- ValueObject.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ValueObject.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/Support/raw_ostream.h"

// Project includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectChild.h"
#include "lldb/Core/ValueObjectList.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include <stdlib.h>

using namespace lldb;
using namespace lldb_private;

static lldb::user_id_t g_value_obj_uid = 0;

//----------------------------------------------------------------------
// ValueObject constructor
//----------------------------------------------------------------------
ValueObject::ValueObject () :
    UserID (++g_value_obj_uid), // Unique identifier for every value object
    m_update_id (0),    // Value object lists always start at 1, value objects start at zero
    m_name (),
    m_data (),
    m_value (),
    m_error (),
    m_flags (),
    m_value_str(),
    m_location_str(),
    m_summary_str(),
    m_children(),
    m_synthetic_children()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ValueObject::~ValueObject ()
{
}

user_id_t
ValueObject::GetUpdateID() const
{
    return m_update_id;
}

bool
ValueObject::UpdateValueIfNeeded (ExecutionContextScope *exe_scope)
{
    if (exe_scope)
    {
        Process *process = exe_scope->CalculateProcess();
        if (process)
        {
            const user_id_t stop_id = process->GetStopID();
            if (m_update_id != stop_id)
            {
                // Save the old value using swap to avoid a string copy which
                // also will clear our m_value_str
                std::string old_value_str;
                old_value_str.swap (m_value_str);
                m_location_str.clear();
                m_summary_str.clear();

                const bool value_was_valid = GetValueIsValid();
                SetValueDidChange (false);

                m_error.Clear();

                // Call the pure virtual function to update the value
                UpdateValue (exe_scope);
                
                // Update the fact that we tried to update the value for this
                // value object wether or not we succeed
                m_update_id = stop_id;
                bool success = m_error.Success();
                SetValueIsValid (success);
                // If the variable hasn't already been marked as changed do it
                // by comparing the old any new value
                if (!GetValueDidChange())
                {
                    if (success)
                    {
                        // The value was gotten successfully, so we consider the
                        // value as changed if the value string differs
                        SetValueDidChange (old_value_str != m_value_str);
                    }
                    else
                    {
                        // The value wasn't gotten successfully, so we mark this
                        // as changed if the value used to be valid and now isn't
                        SetValueDidChange (value_was_valid);
                    }
                }
            }
        }
    }
    return m_error.Success();
}

const DataExtractor &
ValueObject::GetDataExtractor () const
{
    return m_data;
}

DataExtractor &
ValueObject::GetDataExtractor ()
{
    return m_data;
}

const Error &
ValueObject::GetError() const
{
    return m_error;
}

const ConstString &
ValueObject::GetName() const
{
    return m_name;
}

const char *
ValueObject::GetLocationAsCString (ExecutionContextScope *exe_scope)
{
    if (UpdateValueIfNeeded(exe_scope))
    {
        if (m_location_str.empty())
        {
            StreamString sstr;

            switch (m_value.GetValueType())
            {
            default:
                break;

            case Value::eValueTypeScalar:
                if (m_value.GetContextType() == Value::eContextTypeDCRegisterInfo)
                {
                    RegisterInfo *reg_info = m_value.GetRegisterInfo();
                    if (reg_info)
                    {
                        if (reg_info->name)
                            m_location_str = reg_info->name;
                        else if (reg_info->alt_name)
                            m_location_str = reg_info->alt_name;
                        break;
                    }
                }
                m_location_str = "scalar";
                break;

            case Value::eValueTypeLoadAddress:
            case Value::eValueTypeFileAddress:
            case Value::eValueTypeHostAddress:
                {
                    uint32_t addr_nibble_size = m_data.GetAddressByteSize() * 2;
                    sstr.Printf("0x%*.*llx", addr_nibble_size, addr_nibble_size, m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS));
                    m_location_str.swap(sstr.GetString());
                }
                break;
            }
        }
    }
    return m_location_str.c_str();
}

Value &
ValueObject::GetValue()
{
    return m_value;
}

const Value &
ValueObject::GetValue() const
{
    return m_value;
}

bool
ValueObject::GetValueIsValid ()
{
    return m_flags.IsSet(eValueIsValid);
}


void
ValueObject::SetValueIsValid (bool b)
{
    if (b)
        m_flags.Set(eValueIsValid);
    else
        m_flags.Clear(eValueIsValid);
}

bool
ValueObject::GetValueDidChange () const
{
    return m_flags.IsSet(eValueChanged);
}

void
ValueObject::SetValueDidChange (bool value_changed)
{
    m_flags.Set(eValueChanged);
}

ValueObjectSP
ValueObject::GetChildAtIndex (uint32_t idx, bool can_create)
{
    ValueObjectSP child_sp;
    if (idx < GetNumChildren())
    {
        // Check if we have already made the child value object?
        if (can_create && m_children[idx].get() == NULL)
        {
            // No we haven't created the child at this index, so lets have our
            // subclass do it and cache the result for quick future access.
            m_children[idx] = CreateChildAtIndex (idx, false, 0);
        }

        child_sp = m_children[idx];
    }
    return child_sp;
}

uint32_t
ValueObject::GetIndexOfChildWithName (const ConstString &name)
{
    bool omit_empty_base_classes = true;
    return ClangASTContext::GetIndexOfChildWithName (GetClangAST(),
                                                     GetOpaqueClangQualType(),
                                                     name.AsCString(),
                                                     omit_empty_base_classes);
}

ValueObjectSP
ValueObject::GetChildMemberWithName (const ConstString &name, bool can_create)
{
    // when getting a child by name, it could be burried inside some base
    // classes (which really aren't part of the expression path), so we
    // need a vector of indexes that can get us down to the correct child
    std::vector<uint32_t> child_indexes;
    clang::ASTContext *clang_ast = GetClangAST();
    void *clang_type = GetOpaqueClangQualType();
    bool omit_empty_base_classes = true;
    const size_t num_child_indexes =  ClangASTContext::GetIndexOfChildMemberWithName (clang_ast,
                                                                                      clang_type,
                                                                                      name.AsCString(),
                                                                                      omit_empty_base_classes,
                                                                                      child_indexes);
    ValueObjectSP child_sp;
    if (num_child_indexes > 0)
    {
        std::vector<uint32_t>::const_iterator pos = child_indexes.begin ();
        std::vector<uint32_t>::const_iterator end = child_indexes.end ();

        child_sp = GetChildAtIndex(*pos, can_create);
        for (++pos; pos != end; ++pos)
        {
            if (child_sp)
            {
                ValueObjectSP new_child_sp(child_sp->GetChildAtIndex (*pos, can_create));
                child_sp = new_child_sp;
            }
            else
            {
                child_sp.reset();
            }

        }
    }
    return child_sp;
}


uint32_t
ValueObject::GetNumChildren ()
{
    if (m_flags.IsClear(eNumChildrenHasBeenSet))
    {
        SetNumChildren (CalculateNumChildren());
    }
    return m_children.size();
}
void
ValueObject::SetNumChildren (uint32_t num_children)
{
    m_flags.Set(eNumChildrenHasBeenSet);
    m_children.resize(num_children);
}

void
ValueObject::SetName (const char *name)
{
    m_name.SetCString(name);
}

void
ValueObject::SetName (const ConstString &name)
{
    m_name = name;
}

ValueObjectSP
ValueObject::CreateChildAtIndex (uint32_t idx, bool synthetic_array_member, int32_t synthetic_index)
{
    ValueObjectSP valobj_sp;
    bool omit_empty_base_classes = true;

    std::string child_name_str;
    uint32_t child_byte_size = 0;
    int32_t child_byte_offset = 0;
    uint32_t child_bitfield_bit_size = 0;
    uint32_t child_bitfield_bit_offset = 0;
    const bool transparent_pointers = synthetic_array_member == false;
    clang::ASTContext *clang_ast = GetClangAST();
    void *clang_type = GetOpaqueClangQualType();
    void *child_clang_type;
    child_clang_type = ClangASTContext::GetChildClangTypeAtIndex (clang_ast,
                                                                  GetName().AsCString(),
                                                                  clang_type,
                                                                  idx,
                                                                  transparent_pointers,
                                                                  omit_empty_base_classes,
                                                                  child_name_str,
                                                                  child_byte_size,
                                                                  child_byte_offset,
                                                                  child_bitfield_bit_size,
                                                                  child_bitfield_bit_offset);
    if (child_clang_type)
    {
        if (synthetic_index)
            child_byte_offset += child_byte_size * synthetic_index;

        ConstString child_name;
        if (!child_name_str.empty())
            child_name.SetCString (child_name_str.c_str());

        valobj_sp.reset (new ValueObjectChild (this,
                                               clang_ast,
                                               child_clang_type,
                                               child_name,
                                               child_byte_size,
                                               child_byte_offset,
                                               child_bitfield_bit_size,
                                               child_bitfield_bit_offset));
    }
    return valobj_sp;
}

const char *
ValueObject::GetSummaryAsCString (ExecutionContextScope *exe_scope)
{
    if (UpdateValueIfNeeded (exe_scope))
    {
        if (m_summary_str.empty())
        {
            void *clang_type = GetOpaqueClangQualType();

            // See if this is a pointer to a C string?
            uint32_t fixed_length = 0;
            if (clang_type && ClangASTContext::IsCStringType (clang_type, fixed_length))
            {
                Process *process = exe_scope->CalculateProcess();
                if (process != NULL)
                {
                    StreamString sstr;
                    lldb::addr_t cstr_address = LLDB_INVALID_ADDRESS;
                    lldb::AddressType cstr_address_type = eAddressTypeInvalid;
                    switch (GetValue().GetValueType())
                    {
                    case Value::eValueTypeScalar:
                        cstr_address = m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
                        cstr_address_type = eAddressTypeLoad;
                        break;

                    case Value::eValueTypeLoadAddress:
                    case Value::eValueTypeFileAddress:
                    case Value::eValueTypeHostAddress:
                        {
                            uint32_t data_offset = 0;
                            cstr_address = m_data.GetPointer(&data_offset);
                            cstr_address_type = m_value.GetValueAddressType();
                            if (cstr_address_type == eAddressTypeInvalid)
                                cstr_address_type = eAddressTypeLoad;
                        }
                        break;
                    }

                    if (cstr_address != LLDB_INVALID_ADDRESS)
                    {
                        DataExtractor data;
                        size_t bytes_read = 0;
                        std::vector<char> data_buffer;
                        std::vector<char> cstr_buffer;
                        size_t cstr_length;
                        Error error;
                        if (fixed_length > 0)
                        {
                            data_buffer.resize(fixed_length);
                            // Resize the formatted buffer in case every character
                            // uses the "\xXX" format and one extra byte for a NULL
                            cstr_buffer.resize(data_buffer.size() * 4 + 1);
                            data.SetData (&data_buffer.front(), data_buffer.size(), eByteOrderHost);
                            bytes_read = process->ReadMemory (cstr_address, &data_buffer.front(), fixed_length, error);
                            if (bytes_read > 0)
                            {
                                sstr << '"';
                                cstr_length = data.Dump (&sstr,
                                                         0,                 // Start offset in "data"
                                                         eFormatChar,       // Print as characters
                                                         1,                 // Size of item (1 byte for a char!)
                                                         bytes_read,        // How many bytes to print?
                                                         UINT32_MAX,        // num per line
                                                         LLDB_INVALID_ADDRESS,// base address
                                                         0,                 // bitfield bit size
                                                         0);                // bitfield bit offset
                                sstr << '"';
                            }
                        }
                        else
                        {
                            const size_t k_max_buf_size = 256;
                            data_buffer.resize (k_max_buf_size + 1);
                            // NULL terminate in case we don't get the entire C string
                            data_buffer.back() = '\0';
                            // Make a formatted buffer that can contain take 4
                            // bytes per character in case each byte uses the
                            // "\xXX" format and one extra byte for a NULL
                            cstr_buffer.resize (k_max_buf_size * 4 + 1);

                            data.SetData (&data_buffer.front(), data_buffer.size(), eByteOrderHost);
                            size_t total_cstr_len = 0;
                            while ((bytes_read = process->ReadMemory (cstr_address, &data_buffer.front(), k_max_buf_size, error)) > 0)
                            {
                                size_t len = strlen(&data_buffer.front());
                                if (len == 0)
                                    break;
                                if (len > bytes_read)
                                    len = bytes_read;
                                if (sstr.GetSize() == 0)
                                    sstr << '"';

                                cstr_length = data.Dump (&sstr,
                                                         0,                 // Start offset in "data"
                                                         eFormatChar,       // Print as characters
                                                         1,                 // Size of item (1 byte for a char!)
                                                         len,               // How many bytes to print?
                                                         UINT32_MAX,        // num per line
                                                         LLDB_INVALID_ADDRESS,// base address
                                                         0,                 // bitfield bit size
                                                         0);                // bitfield bit offset

                                if (len < k_max_buf_size)
                                    break;
                                cstr_address += total_cstr_len;
                            }
                            if (sstr.GetSize() > 0)
                                sstr << '"';
                        }

                        if (sstr.GetSize() > 0)
                            m_summary_str.assign (sstr.GetData(), sstr.GetSize());
                    }
                }
            }
        }
    }
    if (m_summary_str.empty())
        return NULL;
    return m_summary_str.c_str();
}


const char *
ValueObject::GetValueAsCString (ExecutionContextScope *exe_scope)
{
    // If our byte size is zero this is an aggregate type that has children
    if (ClangASTContext::IsAggregateType (GetOpaqueClangQualType()) == false)
    {
        if (UpdateValueIfNeeded(exe_scope))
        {
            if (m_value_str.empty())
            {
                const Value::ContextType context_type = m_value.GetContextType();

                switch (context_type)
                {
                case Value::eContextTypeOpaqueClangQualType:
                case Value::eContextTypeDCType:
                case Value::eContextTypeDCVariable:
                    {
                        void *clang_type = GetOpaqueClangQualType ();
                        if (clang_type)
                        {
                            StreamString sstr;
                            lldb::Format format = ClangASTType::GetFormat(clang_type);
                            if (ClangASTType::DumpTypeValue(GetClangAST(),            // The clang AST
                                                                clang_type,               // The clang type to display
                                                                &sstr,
                                                                format,                   // Format to display this type with
                                                                m_data,                   // Data to extract from
                                                                0,                        // Byte offset into "m_data"
                                                                GetByteSize(),            // Byte size of item in "m_data"
                                                                GetBitfieldBitSize(),     // Bitfield bit size
                                                                GetBitfieldBitOffset()))  // Bitfield bit offset
                                m_value_str.swap(sstr.GetString());
                            else
                                m_value_str.clear();
                        }
                    }
                    break;

                case Value::eContextTypeDCRegisterInfo:
                    {
                        const RegisterInfo *reg_info = m_value.GetRegisterInfo();
                        if (reg_info)
                        {
                            StreamString reg_sstr;
                            m_data.Dump(&reg_sstr, 0, reg_info->format, reg_info->byte_size, 1, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                            m_value_str.swap(reg_sstr.GetString());
                        }
                    }
                    break;
                    
                default:
                    break;
                }
            }
        }
    }
    if (m_value_str.empty())
        return NULL;
    return m_value_str.c_str();
}

bool
ValueObject::SetValueFromCString (ExecutionContextScope *exe_scope, const char *value_str)
{
    // Make sure our value is up to date first so that our location and location
    // type is valid.
    if (!UpdateValueIfNeeded(exe_scope))
        return false;

    uint32_t count = 0;
    lldb::Encoding encoding = ClangASTType::GetEncoding (GetOpaqueClangQualType(), count);

    char *end = NULL;
    const size_t byte_size = GetByteSize();
    switch (encoding)
    {
    case eEncodingInvalid:
        return false;

    case eEncodingUint:
        if (byte_size > sizeof(unsigned long long))
        {
            return false;
        }
        else
        {
            unsigned long long ull_val = strtoull(value_str, &end, 0);
            if (end && *end != '\0')
                return false;
            m_value = ull_val;
            // Limit the bytes in our m_data appropriately.
            m_value.GetScalar().GetData (m_data, byte_size);
        }
        break;

    case eEncodingSint:
        if (byte_size > sizeof(long long))
        {
            return false;
        }
        else
        {
            long long sll_val = strtoll(value_str, &end, 0);
            if (end && *end != '\0')
                return false;
            m_value = sll_val;
            // Limit the bytes in our m_data appropriately.
            m_value.GetScalar().GetData (m_data, byte_size);
        }
        break;

    case eEncodingIEEE754:
        {
            const off_t byte_offset = GetByteOffset();
            uint8_t *dst = const_cast<uint8_t *>(m_data.PeekData(byte_offset, byte_size));
            if (dst != NULL)
            {
                // We are decoding a float into host byte order below, so make
                // sure m_data knows what it contains.
                m_data.SetByteOrder(eByteOrderHost);
                const size_t converted_byte_size = ClangASTContext::ConvertStringToFloatValue (
                                                        GetClangAST(),
                                                        GetOpaqueClangQualType(),
                                                        value_str,
                                                        dst,
                                                        byte_size);

                if (converted_byte_size == byte_size)
                {
                }
            }
        }
        break;

    case eEncodingVector:
        return false;

    default:
        return false;
    }

    // If we have made it here the value is in m_data and we should write it
    // out to the target
    return Write ();
}

bool
ValueObject::Write ()
{
    // Clear the update ID so the next time we try and read the value
    // we try and read it again.
    m_update_id = 0;

    // TODO: when Value has a method to write a value back, call it from here.
    return false;

}

void
ValueObject::AddSyntheticChild (const ConstString &key, ValueObjectSP& valobj_sp)
{
    m_synthetic_children[key] = valobj_sp;
}

ValueObjectSP
ValueObject::GetSyntheticChild (const ConstString &key) const
{
    ValueObjectSP synthetic_child_sp;
    std::map<ConstString, ValueObjectSP>::const_iterator pos = m_synthetic_children.find (key);
    if (pos != m_synthetic_children.end())
        synthetic_child_sp = pos->second;
    return synthetic_child_sp;
}

bool
ValueObject::IsPointerType ()
{
    return ClangASTContext::IsPointerType (GetOpaqueClangQualType());
}

bool
ValueObject::IsPointerOrReferenceType ()
{
    return ClangASTContext::IsPointerOrReferenceType(GetOpaqueClangQualType());
}

ValueObjectSP
ValueObject::GetSyntheticArrayMemberFromPointer (int32_t index, bool can_create)
{
    ValueObjectSP synthetic_child_sp;
    if (IsPointerType ())
    {
        char index_str[64];
        snprintf(index_str, sizeof(index_str), "[%i]", index);
        ConstString index_const_str(index_str);
        // Check if we have already created a synthetic array member in this
        // valid object. If we have we will re-use it.
        synthetic_child_sp = GetSyntheticChild (index_const_str);
        if (!synthetic_child_sp)
        {
            // We haven't made a synthetic array member for INDEX yet, so
            // lets make one and cache it for any future reference.
            synthetic_child_sp = CreateChildAtIndex(0, true, index);

            // Cache the value if we got one back...
            if (synthetic_child_sp)
                AddSyntheticChild(index_const_str, synthetic_child_sp);
        }
    }
    return synthetic_child_sp;
}
