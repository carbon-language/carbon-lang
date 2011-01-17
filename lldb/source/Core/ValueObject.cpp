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
#include <stdlib.h>

// C++ Includes
// Other libraries and framework includes
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/Type.h"

// Project includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectChild.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectList.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

static lldb::user_id_t g_value_obj_uid = 0;

//----------------------------------------------------------------------
// ValueObject constructor
//----------------------------------------------------------------------
ValueObject::ValueObject (ValueObject *parent) :
    UserID (++g_value_obj_uid), // Unique identifier for every value object
    m_parent (parent),
    m_update_id (0),    // Value object lists always start at 1, value objects start at zero
    m_name (),
    m_data (),
    m_value (),
    m_error (),
    m_value_str (),
    m_old_value_str (),
    m_location_str (),
    m_summary_str (),
    m_object_desc_str (),
    m_children (),
    m_synthetic_children (),
    m_dynamic_value_sp (),
    m_format (eFormatDefault),
    m_value_is_valid (false),
    m_value_did_change (false),
    m_children_count_valid (false),
    m_old_value_valid (false),
    m_pointers_point_to_load_addrs (false)
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
    // If this is a constant value, then our success is predicated on whether
    // we have an error or not
    if (GetIsConstant())
        return m_error.Success();

    if (exe_scope)
    {
        Process *process = exe_scope->CalculateProcess();
        if (process)
        {
            const user_id_t stop_id = process->GetStopID();
            if (m_update_id != stop_id)
            {
                bool first_update = m_update_id == 0;
                // Save the old value using swap to avoid a string copy which
                // also will clear our m_value_str
                if (m_value_str.empty())
                {
                    m_old_value_valid = false;
                }
                else
                {
                    m_old_value_valid = true;
                    m_old_value_str.swap (m_value_str);
                    m_value_str.clear();
                }
                m_location_str.clear();
                m_summary_str.clear();
                m_object_desc_str.clear();

                const bool value_was_valid = GetValueIsValid();
                SetValueDidChange (false);

                m_error.Clear();

                // Call the pure virtual function to update the value
                UpdateValue (exe_scope);
                
                // Update the fact that we tried to update the value for this
                // value object whether or not we succeed
                m_update_id = stop_id;
                bool success = m_error.Success();
                SetValueIsValid (success);
                
                if (first_update)
                    SetValueDidChange (false);
                else if (!m_value_did_change && success == false)
                {
                    // The value wasn't gotten successfully, so we mark this
                    // as changed if the value used to be valid and now isn't
                    SetValueDidChange (value_was_valid);
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
                if (m_value.GetContextType() == Value::eContextTypeRegisterInfo)
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
ValueObject::ResolveValue (ExecutionContextScope *exe_scope, Scalar &scalar)
{
    ExecutionContext exe_ctx;
    exe_scope->CalculateExecutionContext(exe_ctx);
    scalar = m_value.ResolveValue(&exe_ctx, GetClangAST ());
    return scalar.IsValid();
}

bool
ValueObject::GetValueIsValid () const
{
    return m_value_is_valid;
}


void
ValueObject::SetValueIsValid (bool b)
{
    m_value_is_valid = b;
}

bool
ValueObject::GetValueDidChange (ExecutionContextScope *exe_scope)
{
    GetValueAsCString (exe_scope);
    return m_value_did_change;
}

void
ValueObject::SetValueDidChange (bool value_changed)
{
    m_value_did_change = value_changed;
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
                                                     GetClangType(),
                                                     name.GetCString(),
                                                     omit_empty_base_classes);
}

ValueObjectSP
ValueObject::GetChildMemberWithName (const ConstString &name, bool can_create)
{
    // when getting a child by name, it could be buried inside some base
    // classes (which really aren't part of the expression path), so we
    // need a vector of indexes that can get us down to the correct child
    std::vector<uint32_t> child_indexes;
    clang::ASTContext *clang_ast = GetClangAST();
    void *clang_type = GetClangType();
    bool omit_empty_base_classes = true;
    const size_t num_child_indexes =  ClangASTContext::GetIndexOfChildMemberWithName (clang_ast,
                                                                                      clang_type,
                                                                                      name.GetCString(),
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
    if (!m_children_count_valid)
    {
        SetNumChildren (CalculateNumChildren());
    }
    return m_children.size();
}
void
ValueObject::SetNumChildren (uint32_t num_children)
{
    m_children_count_valid = true;
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
    bool child_is_base_class = false;
    const bool transparent_pointers = synthetic_array_member == false;
    clang::ASTContext *clang_ast = GetClangAST();
    clang_type_t clang_type = GetClangType();
    clang_type_t child_clang_type;
    child_clang_type = ClangASTContext::GetChildClangTypeAtIndex (clang_ast,
                                                                  GetName().GetCString(),
                                                                  clang_type,
                                                                  idx,
                                                                  transparent_pointers,
                                                                  omit_empty_base_classes,
                                                                  child_name_str,
                                                                  child_byte_size,
                                                                  child_byte_offset,
                                                                  child_bitfield_bit_size,
                                                                  child_bitfield_bit_offset,
                                                                  child_is_base_class);
    if (child_clang_type && child_byte_size)
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
                                               child_bitfield_bit_offset,
                                               child_is_base_class));
        if (m_pointers_point_to_load_addrs)
            valobj_sp->SetPointersPointToLoadAddrs (m_pointers_point_to_load_addrs);
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
            clang_type_t clang_type = GetClangType();

            // See if this is a pointer to a C string?
            if (clang_type)
            {
                StreamString sstr;
                clang_type_t elem_or_pointee_clang_type;
                const Flags type_flags (ClangASTContext::GetTypeInfo (clang_type, 
                                                                      GetClangAST(), 
                                                                      &elem_or_pointee_clang_type));

                if (type_flags.AnySet (ClangASTContext::eTypeIsArray | ClangASTContext::eTypeIsPointer) &&
                    ClangASTContext::IsCharType (elem_or_pointee_clang_type))
                {
                    Process *process = exe_scope->CalculateProcess();
                    if (process != NULL)
                    {
                        lldb::addr_t cstr_address = LLDB_INVALID_ADDRESS;
                        lldb::AddressType cstr_address_type = eAddressTypeInvalid;

                        size_t cstr_len = 0;
                        if (type_flags.Test (ClangASTContext::eTypeIsArray))
                        {
                            // We have an array
                            cstr_len = ClangASTContext::GetArraySize (clang_type);
                            cstr_address = GetAddressOf (cstr_address_type, true);
                        }
                        else
                        {
                            // We have a pointer
                            cstr_address = GetPointerValue (cstr_address_type, true);
                        }
                        if (cstr_address != LLDB_INVALID_ADDRESS)
                        {
                            DataExtractor data;
                            size_t bytes_read = 0;
                            std::vector<char> data_buffer;
                            std::vector<char> cstr_buffer;
                            size_t cstr_length;
                            Error error;
                            if (cstr_len > 0)
                            {
                                data_buffer.resize(cstr_len);
                                // Resize the formatted buffer in case every character
                                // uses the "\xXX" format and one extra byte for a NULL
                                cstr_buffer.resize(data_buffer.size() * 4 + 1);
                                data.SetData (&data_buffer.front(), data_buffer.size(), eByteOrderHost);
                                bytes_read = process->ReadMemory (cstr_address, &data_buffer.front(), cstr_len, error);
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
                        }
                    }
                    
                    if (sstr.GetSize() > 0)
                        m_summary_str.assign (sstr.GetData(), sstr.GetSize());
                }
                else if (ClangASTContext::IsFunctionPointerType (clang_type))
                {
                    lldb::AddressType func_ptr_address_type = eAddressTypeInvalid;
                    lldb::addr_t func_ptr_address = GetPointerValue (func_ptr_address_type, true);

                    if (func_ptr_address != 0 && func_ptr_address != LLDB_INVALID_ADDRESS)
                    {
                        switch (func_ptr_address_type)
                        {
                        case eAddressTypeInvalid:
                        case eAddressTypeFile:
                            break;

                        case eAddressTypeLoad:
                            {
                                Address so_addr;
                                Target *target = exe_scope->CalculateTarget();
                                if (target && target->GetSectionLoadList().IsEmpty() == false)
                                {
                                    if (target->GetSectionLoadList().ResolveLoadAddress(func_ptr_address, so_addr))
                                    {
                                        so_addr.Dump (&sstr, 
                                                      exe_scope, 
                                                      Address::DumpStyleResolvedDescription, 
                                                      Address::DumpStyleSectionNameOffset);
                                    }
                                }
                            }
                            break;

                        case eAddressTypeHost:
                            break;
                        }
                    }
                    if (sstr.GetSize() > 0)
                    {
                        m_summary_str.assign (1, '(');
                        m_summary_str.append (sstr.GetData(), sstr.GetSize());
                        m_summary_str.append (1, ')');
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
ValueObject::GetObjectDescription (ExecutionContextScope *exe_scope)
{
    if (!m_object_desc_str.empty())
        return m_object_desc_str.c_str();
        
    if (!GetValueIsValid())
        return NULL;
    
    Process *process = exe_scope->CalculateProcess();
    if (process == NULL)
        return NULL;
        
    StreamString s;
    
    lldb::LanguageType language = GetObjectRuntimeLanguage();
    LanguageRuntime *runtime = process->GetLanguageRuntime(language);
    
    if (runtime == NULL)
    {
        // Aw, hell, if the things a pointer, let's try ObjC anyway...
        clang_type_t opaque_qual_type = GetClangType();
        if (opaque_qual_type != NULL)
        {
            clang::QualType qual_type (clang::QualType::getFromOpaquePtr(opaque_qual_type).getNonReferenceType());
            if (qual_type->isAnyPointerType())
                runtime = process->GetLanguageRuntime(lldb::eLanguageTypeObjC);
        }
    }
    
    if (runtime && runtime->GetObjectDescription(s, *this, exe_scope))
    {
        m_object_desc_str.append (s.GetData());
    }
    
    if (m_object_desc_str.empty())
        return NULL;
    else
        return m_object_desc_str.c_str();
}

const char *
ValueObject::GetValueAsCString (ExecutionContextScope *exe_scope)
{
    // If our byte size is zero this is an aggregate type that has children
    if (ClangASTContext::IsAggregateType (GetClangType()) == false)
    {
        if (UpdateValueIfNeeded(exe_scope))
        {
            if (m_value_str.empty())
            {
                const Value::ContextType context_type = m_value.GetContextType();

                switch (context_type)
                {
                case Value::eContextTypeClangType:
                case Value::eContextTypeLLDBType:
                case Value::eContextTypeVariable:
                    {
                        clang_type_t clang_type = GetClangType ();
                        if (clang_type)
                        {
                            StreamString sstr;
                            if (m_format == eFormatDefault)
                                m_format = ClangASTType::GetFormat(clang_type);

                            if (ClangASTType::DumpTypeValue (GetClangAST(),            // The clang AST
                                                             clang_type,               // The clang type to display
                                                             &sstr,
                                                             m_format,                 // Format to display this type with
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

                case Value::eContextTypeRegisterInfo:
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
            
            if (!m_value_did_change && m_old_value_valid)
            {
                // The value was gotten successfully, so we consider the
                // value as changed if the value string differs
                SetValueDidChange (m_old_value_str != m_value_str);
            }
        }
    }
    if (m_value_str.empty())
        return NULL;
    return m_value_str.c_str();
}

addr_t
ValueObject::GetAddressOf (lldb::AddressType &address_type, bool scalar_is_load_address)
{
    switch (m_value.GetValueType())
    {
    case Value::eValueTypeScalar:
        if (scalar_is_load_address)
        {
            address_type = eAddressTypeLoad;
            return m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
        }
        break;

    case Value::eValueTypeLoadAddress: 
    case Value::eValueTypeFileAddress:
    case Value::eValueTypeHostAddress:
        {
            address_type = m_value.GetValueAddressType ();
            return m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
        }
        break;
    }
    address_type = eAddressTypeInvalid;
    return LLDB_INVALID_ADDRESS;
}

addr_t
ValueObject::GetPointerValue (lldb::AddressType &address_type, bool scalar_is_load_address)
{
    lldb::addr_t address = LLDB_INVALID_ADDRESS;
    address_type = eAddressTypeInvalid;
    switch (m_value.GetValueType())
    {
    case Value::eValueTypeScalar:
        if (scalar_is_load_address)
        {
            address = m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
            address_type = eAddressTypeLoad;
        }
        break;

    case Value::eValueTypeLoadAddress:
    case Value::eValueTypeFileAddress:
    case Value::eValueTypeHostAddress:
        {
            uint32_t data_offset = 0;
            address = m_data.GetPointer(&data_offset);
            address_type = m_value.GetValueAddressType();
            if (address_type == eAddressTypeInvalid)
                address_type = eAddressTypeLoad;
        }
        break;
    }

    if (m_pointers_point_to_load_addrs)
        address_type = eAddressTypeLoad;

    return address;
}

bool
ValueObject::SetValueFromCString (ExecutionContextScope *exe_scope, const char *value_str)
{
    // Make sure our value is up to date first so that our location and location
    // type is valid.
    if (!UpdateValueIfNeeded(exe_scope))
        return false;

    uint32_t count = 0;
    lldb::Encoding encoding = ClangASTType::GetEncoding (GetClangType(), count);

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
                                                        GetClangType(),
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

lldb::LanguageType
ValueObject::GetObjectRuntimeLanguage ()
{
    clang_type_t opaque_qual_type = GetClangType();
    if (opaque_qual_type == NULL)
        return lldb::eLanguageTypeC;
    
    // If the type is a reference, then resolve it to what it refers to first:     
    clang::QualType qual_type (clang::QualType::getFromOpaquePtr(opaque_qual_type).getNonReferenceType());
    if (qual_type->isAnyPointerType())
    {
        if (qual_type->isObjCObjectPointerType())
            return lldb::eLanguageTypeObjC;

        clang::QualType pointee_type (qual_type->getPointeeType());
        if (pointee_type->getCXXRecordDeclForPointerType() != NULL)
            return lldb::eLanguageTypeC_plus_plus;
        if (pointee_type->isObjCObjectOrInterfaceType())
            return lldb::eLanguageTypeObjC;
        if (pointee_type->isObjCClassType())
            return lldb::eLanguageTypeObjC;
    }
    else
    {
        if (ClangASTContext::IsObjCClassType (opaque_qual_type))
            return lldb::eLanguageTypeObjC;
        if (ClangASTContext::IsCXXClassType (opaque_qual_type))
            return lldb::eLanguageTypeC_plus_plus;
    }
            
    return lldb::eLanguageTypeC;
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
    return ClangASTContext::IsPointerType (GetClangType());
}



bool
ValueObject::IsPointerOrReferenceType ()
{
    return ClangASTContext::IsPointerOrReferenceType(GetClangType());
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

bool
ValueObject::SetDynamicValue ()
{
    if (!IsPointerOrReferenceType())
        return false;
        
    // Check that the runtime class is correct for determining the most specific class.
    // If it is a C++ class, see if it is dynamic:
    
    return true;
}


void
ValueObject::GetExpressionPath (Stream &s, bool qualify_cxx_base_classes)
{
    if (m_parent)
    {
        m_parent->GetExpressionPath (s, qualify_cxx_base_classes);
        clang_type_t parent_clang_type = m_parent->GetClangType();
        if (parent_clang_type)
        {
            if (ClangASTContext::IsPointerType(parent_clang_type))
            {
                s.PutCString("->");
            }
            else if (ClangASTContext::IsAggregateType (parent_clang_type))
            {
                if (ClangASTContext::IsArrayType (parent_clang_type) == false &&
                    m_parent->IsBaseClass() == false)
                    s.PutChar('.');
            }
        }
    }
    
    if (IsBaseClass())
    {
        if (qualify_cxx_base_classes)
        {
            clang_type_t clang_type = GetClangType();
            std::string cxx_class_name;
            if (ClangASTContext::GetCXXClassName (clang_type, cxx_class_name))
            {
                s << cxx_class_name.c_str() << "::";
            }
        }
    }
    else
    {
        const char *name = GetName().GetCString();
        if (name)
            s.PutCString(name);    
    }
}

void
ValueObject::DumpValueObject 
(
    Stream &s,
    ExecutionContextScope *exe_scope,
    ValueObject *valobj,
    const char *root_valobj_name,
    uint32_t ptr_depth,
    uint32_t curr_depth,
    uint32_t max_depth,
    bool show_types,
    bool show_location,
    bool use_objc,
    bool scope_already_checked,
    bool flat_output
)
{
    if (valobj)
    {
        clang_type_t clang_type = valobj->GetClangType();

        const Flags type_flags (ClangASTContext::GetTypeInfo (clang_type, NULL, NULL));
        const char *err_cstr = NULL;
        const bool has_children = type_flags.Test (ClangASTContext::eTypeHasChildren);
        const bool has_value = type_flags.Test (ClangASTContext::eTypeHasValue);
        
        const bool print_valobj = flat_output == false || has_value;
        
        if (print_valobj)
        {
            if (show_location)
            {
                s.Printf("%s: ", valobj->GetLocationAsCString(exe_scope));
            }

            s.Indent();

            // Always show the type for the top level items.
            if (show_types || curr_depth == 0)
                s.Printf("(%s) ", valobj->GetTypeName().AsCString("<invalid type>"));


            if (flat_output)
            {
                // If we are showing types, also qualify the C++ base classes 
                const bool qualify_cxx_base_classes = show_types;
                valobj->GetExpressionPath(s, qualify_cxx_base_classes);
                s.PutCString(" =");
            }
            else
            {
                const char *name_cstr = root_valobj_name ? root_valobj_name : valobj->GetName().AsCString("");
                s.Printf ("%s =", name_cstr);
            }

            if (!scope_already_checked && !valobj->IsInScope(exe_scope->CalculateStackFrame()))
            {
                err_cstr = "error: out of scope";
            }
        }
        
        const char *val_cstr = NULL;
        
        if (err_cstr == NULL)
        {
            val_cstr = valobj->GetValueAsCString(exe_scope);
            err_cstr = valobj->GetError().AsCString();
        }

        if (err_cstr)
        {
            s.Printf (" error: %s\n", err_cstr);
        }
        else
        {
            const bool is_ref = type_flags.Test (ClangASTContext::eTypeIsReference);
            if (print_valobj)
            {
                const char *sum_cstr = valobj->GetSummaryAsCString(exe_scope);

                if (val_cstr)
                    s.Printf(" %s", val_cstr);

                if (sum_cstr)
                    s.Printf(" %s", sum_cstr);
                
                if (use_objc)
                {
                    const char *object_desc = valobj->GetObjectDescription(exe_scope);
                    if (object_desc)
                        s.Printf(" %s\n", object_desc);
                    else
                        s.Printf (" [no Objective-C description available]\n");
                    return;
                }                
            }

            if (curr_depth < max_depth)
            {
                // We will show children for all concrete types. We won't show
                // pointer contents unless a pointer depth has been specified.
                // We won't reference contents unless the reference is the 
                // root object (depth of zero).
                bool print_children = true;

                // Use a new temporary pointer depth in case we override the
                // current pointer depth below...
                uint32_t curr_ptr_depth = ptr_depth;

                const bool is_ptr = type_flags.Test (ClangASTContext::eTypeIsPointer);
                if (is_ptr || is_ref)
                {
                    // We have a pointer or reference whose value is an address.
                    // Make sure that address is not NULL
                    lldb::AddressType ptr_address_type;
                    if (valobj->GetPointerValue (ptr_address_type, true) == 0)
                        print_children = false;

                    else if (is_ref && curr_depth == 0)
                    {
                        // If this is the root object (depth is zero) that we are showing
                        // and it is a reference, and no pointer depth has been supplied
                        // print out what it references. Don't do this at deeper depths
                        // otherwise we can end up with infinite recursion...
                        curr_ptr_depth = 1;
                    }
                    
                    if (curr_ptr_depth == 0)
                        print_children = false;
                }
                
                if (print_children)
                {
                    const uint32_t num_children = valobj->GetNumChildren();
                    if (num_children)
                    {
                        if (flat_output)
                        {
                            if (print_valobj)
                                s.EOL();
                        }
                        else
                        {
                            if (print_valobj)
                                s.PutCString(is_ref ? ": {\n" : " {\n");
                            s.IndentMore();
                        }

                        for (uint32_t idx=0; idx<num_children; ++idx)
                        {
                            ValueObjectSP child_sp(valobj->GetChildAtIndex(idx, true));
                            if (child_sp.get())
                            {
                                DumpValueObject (s,
                                                 exe_scope,
                                                 child_sp.get(),
                                                 NULL,
                                                 (is_ptr || is_ref) ? curr_ptr_depth - 1 : curr_ptr_depth,
                                                 curr_depth + 1,
                                                 max_depth,
                                                 show_types,
                                                 show_location,
                                                 false,
                                                 true,
                                                 flat_output);
                            }
                        }

                        if (!flat_output)
                        {
                            s.IndentLess();
                            s.Indent("}\n");
                        }
                    }
                    else if (has_children)
                    {
                        // Aggregate, no children...
                        if (print_valobj)
                            s.PutCString(" {}\n");
                    }
                    else
                    {
                        if (print_valobj)
                            s.EOL();
                    }

                }
                else
                {  
                    s.EOL();
                }
            }
            else
            {
                if (has_children && print_valobj)
                {
                    s.PutCString("{...}\n");
                }
            }
        }
    }
}


ValueObjectSP
ValueObject::CreateConstantValue (ExecutionContextScope *exe_scope, const ConstString &name)
{
    ValueObjectSP valobj_sp;
    
    if (UpdateValueIfNeeded(exe_scope) && m_error.Success())
    {
        ExecutionContext exe_ctx;
        exe_scope->CalculateExecutionContext(exe_ctx);

        clang::ASTContext *ast = GetClangAST ();

        DataExtractor data;
        data.SetByteOrder (m_data.GetByteOrder());
        data.SetAddressByteSize(m_data.GetAddressByteSize());

        m_error = m_value.GetValueAsData (&exe_ctx, ast, data, 0);

        valobj_sp.reset (new ValueObjectConstResult (ast,
                                                     GetClangType(),
                                                     name,
                                                     data));
    }
    else
    {
        valobj_sp.reset (new ValueObjectConstResult (m_error));
    }
    return valobj_sp;
}

lldb::ValueObjectSP
ValueObject::Dereference (Error &error)
{
    lldb::ValueObjectSP valobj_sp;
    const bool is_pointer_type = IsPointerType();
    if (is_pointer_type)
    {
        bool omit_empty_base_classes = true;

        std::string child_name_str;
        uint32_t child_byte_size = 0;
        int32_t child_byte_offset = 0;
        uint32_t child_bitfield_bit_size = 0;
        uint32_t child_bitfield_bit_offset = 0;
        bool child_is_base_class = false;
        const bool transparent_pointers = false;
        clang::ASTContext *clang_ast = GetClangAST();
        clang_type_t clang_type = GetClangType();
        clang_type_t child_clang_type;
        child_clang_type = ClangASTContext::GetChildClangTypeAtIndex (clang_ast,
                                                                      GetName().GetCString(),
                                                                      clang_type,
                                                                      0,
                                                                      transparent_pointers,
                                                                      omit_empty_base_classes,
                                                                      child_name_str,
                                                                      child_byte_size,
                                                                      child_byte_offset,
                                                                      child_bitfield_bit_size,
                                                                      child_bitfield_bit_offset,
                                                                      child_is_base_class);
        if (child_clang_type && child_byte_size)
        {
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
                                                   child_bitfield_bit_offset,
                                                   child_is_base_class));
        }
    }

    if (valobj_sp)
    {
        error.Clear();
    }
    else
    {
        StreamString strm;
        GetExpressionPath(strm, true);

        if (is_pointer_type)
            error.SetErrorStringWithFormat("dereference failed: (%s) %s", GetTypeName().AsCString("<invalid type>"), strm.GetString().c_str());
        else
            error.SetErrorStringWithFormat("not a pointer type: (%s) %s", GetTypeName().AsCString("<invalid type>"), strm.GetString().c_str());
    }

    return valobj_sp;
}

    lldb::ValueObjectSP
ValueObject::AddressOf (Error &error)
{
    lldb::ValueObjectSP valobj_sp;
    lldb::AddressType address_type = eAddressTypeInvalid;
    const bool scalar_is_load_address = false;
    lldb::addr_t addr = GetAddressOf (address_type, scalar_is_load_address);
    error.Clear();
    if (addr != LLDB_INVALID_ADDRESS)
    {
        switch (address_type)
        {
        default:
        case eAddressTypeInvalid:
            {
                StreamString expr_path_strm;
                GetExpressionPath(expr_path_strm, true);
                error.SetErrorStringWithFormat("'%s' is not in memory", expr_path_strm.GetString().c_str());
            }
            break;

        case eAddressTypeFile:
        case eAddressTypeLoad:
        case eAddressTypeHost:
            {
                clang::ASTContext *ast = GetClangAST();
                clang_type_t clang_type = GetClangType();
                if (ast && clang_type)
                {
                    std::string name (1, '&');
                    name.append (m_name.AsCString(""));
                    valobj_sp.reset (new ValueObjectConstResult (ast, 
                                                                 ClangASTContext::CreatePointerType (ast, clang_type),
                                                                 ConstString (name.c_str()),
                                                                 addr, 
                                                                 eAddressTypeInvalid,
                                                                 m_data.GetAddressByteSize()));
                }
            }
            break;
        }
    }
    return valobj_sp;
}

