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
#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectChild.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectDynamicValue.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Core/ValueObjectSyntheticFilter.h"

#include "lldb/Host/Endian.h"

#include "lldb/Interpreter/ScriptInterpreterPython.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "lldb/Utility/RefCounter.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_utility;

static lldb::user_id_t g_value_obj_uid = 0;

//----------------------------------------------------------------------
// ValueObject constructor
//----------------------------------------------------------------------
ValueObject::ValueObject (ValueObject &parent) :
    UserID (++g_value_obj_uid), // Unique identifier for every value object
    m_parent (&parent),
    m_update_point (parent.GetUpdatePoint ()),
    m_name (),
    m_data (),
    m_value (),
    m_error (),
    m_value_str (),
    m_old_value_str (),
    m_location_str (),
    m_summary_str (),
    m_object_desc_str (),
    m_manager(parent.GetManager()),
    m_children (),
    m_synthetic_children (),
    m_dynamic_value (NULL),
    m_synthetic_value(NULL),
    m_deref_valobj(NULL),
    m_format (eFormatDefault),
    m_last_format_mgr_revision(0),
    m_last_summary_format(),
    m_forced_summary_format(),
    m_last_value_format(),
    m_last_synthetic_filter(),
    m_user_id_of_forced_summary(0),
    m_value_is_valid (false),
    m_value_did_change (false),
    m_children_count_valid (false),
    m_old_value_valid (false),
    m_pointers_point_to_load_addrs (false),
    m_is_deref_of_parent (false),
    m_is_array_item_for_pointer(false),
    m_is_bitfield_for_scalar(false),
    m_is_expression_path_child(false),
    m_dump_printable_counter(0)
{
    m_manager->ManageObject(this);
}

//----------------------------------------------------------------------
// ValueObject constructor
//----------------------------------------------------------------------
ValueObject::ValueObject (ExecutionContextScope *exe_scope) :
    UserID (++g_value_obj_uid), // Unique identifier for every value object
    m_parent (NULL),
    m_update_point (exe_scope),
    m_name (),
    m_data (),
    m_value (),
    m_error (),
    m_value_str (),
    m_old_value_str (),
    m_location_str (),
    m_summary_str (),
    m_object_desc_str (),
    m_manager(),
    m_children (),
    m_synthetic_children (),
    m_dynamic_value (NULL),
    m_synthetic_value(NULL),
    m_deref_valobj(NULL),
    m_format (eFormatDefault),
    m_last_format_mgr_revision(0),
    m_last_summary_format(),
    m_forced_summary_format(),
    m_last_value_format(),
    m_last_synthetic_filter(),
    m_user_id_of_forced_summary(0),
    m_value_is_valid (false),
    m_value_did_change (false),
    m_children_count_valid (false),
    m_old_value_valid (false),
    m_pointers_point_to_load_addrs (false),
    m_is_deref_of_parent (false),
    m_is_array_item_for_pointer(false),
    m_is_bitfield_for_scalar(false),
    m_is_expression_path_child(false),
    m_dump_printable_counter(0)
{
    m_manager = new ValueObjectManager();
    m_manager->ManageObject (this);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ValueObject::~ValueObject ()
{
}

bool
ValueObject::UpdateValueIfNeeded (bool update_format)
{
    
    if (update_format)
        UpdateFormatsIfNeeded();
    
    // If this is a constant value, then our success is predicated on whether
    // we have an error or not
    if (GetIsConstant())
        return m_error.Success();

    bool first_update = m_update_point.IsFirstEvaluation();
    
    if (m_update_point.NeedsUpdating())
    {
        m_update_point.SetUpdated();
        
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

        ClearUserVisibleData();
        
        const bool value_was_valid = GetValueIsValid();
        SetValueDidChange (false);

        m_error.Clear();

        // Call the pure virtual function to update the value
        bool success = UpdateValue ();
        
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
    return m_error.Success();
}

void
ValueObject::UpdateFormatsIfNeeded()
{
    /*printf("CHECKING FOR UPDATES. I am at revision %d, while the format manager is at revision %d\n",
           m_last_format_mgr_revision,
           Debugger::ValueFormats::GetCurrentRevision());*/
    if (HasCustomSummaryFormat() && m_update_point.GetUpdateID() != m_user_id_of_forced_summary)
    {
        ClearCustomSummaryFormat();
        m_summary_str.clear();
    }
    if (m_last_format_mgr_revision != Debugger::Formatting::ValueFormats::GetCurrentRevision())
    {
        if (m_last_summary_format.get())
            m_last_summary_format.reset((StringSummaryFormat*)NULL);
        if (m_last_value_format.get())
            m_last_value_format.reset(/*(ValueFormat*)NULL*/);
        if (m_last_synthetic_filter.get())
            m_last_synthetic_filter.reset(/*(SyntheticFilter*)NULL*/);

        m_synthetic_value = NULL;
        
        Debugger::Formatting::ValueFormats::Get(*this, m_last_value_format);
        Debugger::Formatting::GetSummaryFormat(*this, m_last_summary_format);
        Debugger::Formatting::GetSyntheticFilter(*this, m_last_synthetic_filter);

        m_last_format_mgr_revision = Debugger::Formatting::ValueFormats::GetCurrentRevision();

        ClearUserVisibleData();
    }
}

DataExtractor &
ValueObject::GetDataExtractor ()
{
    UpdateValueIfNeeded();
    return m_data;
}

const Error &
ValueObject::GetError()
{
    UpdateValueIfNeeded();
    return m_error;
}

const ConstString &
ValueObject::GetName() const
{
    return m_name;
}

const char *
ValueObject::GetLocationAsCString ()
{
    if (UpdateValueIfNeeded())
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
ValueObject::ResolveValue (Scalar &scalar)
{
    ExecutionContext exe_ctx;
    ExecutionContextScope *exe_scope = GetExecutionContextScope();
    if (exe_scope)
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
ValueObject::GetValueDidChange ()
{
    GetValueAsCString ();
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
    // We may need to update our value if we are dynamic
    if (IsPossibleDynamicType ())
        UpdateValueIfNeeded();
    if (idx < GetNumChildren())
    {
        // Check if we have already made the child value object?
        if (can_create && m_children[idx] == NULL)
        {
            // No we haven't created the child at this index, so lets have our
            // subclass do it and cache the result for quick future access.
            m_children[idx] = CreateChildAtIndex (idx, false, 0);
        }
        
        if (m_children[idx] != NULL)
            return m_children[idx]->GetSP();
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
    ValueObjectSP child_sp;

    // We may need to update our value if we are dynamic
    if (IsPossibleDynamicType ())
        UpdateValueIfNeeded();

    std::vector<uint32_t> child_indexes;
    clang::ASTContext *clang_ast = GetClangAST();
    void *clang_type = GetClangType();
    bool omit_empty_base_classes = true;
    const size_t num_child_indexes =  ClangASTContext::GetIndexOfChildMemberWithName (clang_ast,
                                                                                      clang_type,
                                                                                      name.GetCString(),
                                                                                      omit_empty_base_classes,
                                                                                      child_indexes);
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

ValueObject *
ValueObject::CreateChildAtIndex (uint32_t idx, bool synthetic_array_member, int32_t synthetic_index)
{
    ValueObject *valobj = NULL;
    
    bool omit_empty_base_classes = true;
    bool ignore_array_bounds = synthetic_array_member;
    std::string child_name_str;
    uint32_t child_byte_size = 0;
    int32_t child_byte_offset = 0;
    uint32_t child_bitfield_bit_size = 0;
    uint32_t child_bitfield_bit_offset = 0;
    bool child_is_base_class = false;
    bool child_is_deref_of_parent = false;

    const bool transparent_pointers = synthetic_array_member == false;
    clang::ASTContext *clang_ast = GetClangAST();
    clang_type_t clang_type = GetClangType();
    clang_type_t child_clang_type;
    
    ExecutionContext exe_ctx;
    GetExecutionContextScope()->CalculateExecutionContext (exe_ctx);
    
    child_clang_type = ClangASTContext::GetChildClangTypeAtIndex (&exe_ctx,
                                                                  clang_ast,
                                                                  GetName().GetCString(),
                                                                  clang_type,
                                                                  idx,
                                                                  transparent_pointers,
                                                                  omit_empty_base_classes,
                                                                  ignore_array_bounds,
                                                                  child_name_str,
                                                                  child_byte_size,
                                                                  child_byte_offset,
                                                                  child_bitfield_bit_size,
                                                                  child_bitfield_bit_offset,
                                                                  child_is_base_class,
                                                                  child_is_deref_of_parent);
    if (child_clang_type && child_byte_size)
    {
        if (synthetic_index)
            child_byte_offset += child_byte_size * synthetic_index;

        ConstString child_name;
        if (!child_name_str.empty())
            child_name.SetCString (child_name_str.c_str());

        valobj = new ValueObjectChild (*this,
                                       clang_ast,
                                       child_clang_type,
                                       child_name,
                                       child_byte_size,
                                       child_byte_offset,
                                       child_bitfield_bit_size,
                                       child_bitfield_bit_offset,
                                       child_is_base_class,
                                       child_is_deref_of_parent);            
        if (m_pointers_point_to_load_addrs)
            valobj->SetPointersPointToLoadAddrs (m_pointers_point_to_load_addrs);
    }
    
    return valobj;
}

const char *
ValueObject::GetSummaryAsCString ()
{
    if (UpdateValueIfNeeded ())
    {        
        if (m_summary_str.empty())
        {
            SummaryFormat *summary_format = GetSummaryFormat().get();

            if (summary_format)
            {
                m_summary_str = summary_format->FormatObject(GetSP());
            }
            else
            {
                clang_type_t clang_type = GetClangType();

                // Do some default printout for function pointers
                if (clang_type)
                {
                    StreamString sstr;
                    clang_type_t elem_or_pointee_clang_type;
                    const Flags type_flags (ClangASTContext::GetTypeInfo (clang_type, 
                                                                          GetClangAST(), 
                                                                          &elem_or_pointee_clang_type));

                    ExecutionContextScope *exe_scope = GetExecutionContextScope();
                    if (exe_scope)
                    {
                        if (ClangASTContext::IsFunctionPointerType (clang_type))
                        {
                            AddressType func_ptr_address_type = eAddressTypeInvalid;
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
        }
    }
    if (m_summary_str.empty())
        return NULL;
    return m_summary_str.c_str();
}

bool
ValueObject::IsCStringContainer(bool check_pointer)
{
    clang_type_t elem_or_pointee_clang_type;
    const Flags type_flags (ClangASTContext::GetTypeInfo (GetClangType(), 
                                                          GetClangAST(), 
                                                          &elem_or_pointee_clang_type));
    bool is_char_arr_ptr (type_flags.AnySet (ClangASTContext::eTypeIsArray | ClangASTContext::eTypeIsPointer) &&
            ClangASTContext::IsCharType (elem_or_pointee_clang_type));
    if (!is_char_arr_ptr)
        return false;
    if (!check_pointer)
        return true;
    if (type_flags.Test(ClangASTContext::eTypeIsArray))
        return true;
    lldb::addr_t cstr_address = LLDB_INVALID_ADDRESS;
    AddressType cstr_address_type = eAddressTypeInvalid;
    cstr_address = GetAddressOf (cstr_address_type, true);
    return (cstr_address != LLDB_INVALID_ADDRESS);
}

void
ValueObject::ReadPointedString(Stream& s,
                               Error& error,
                               uint32_t max_length,
                               bool honor_array,
                               lldb::Format item_format)
{
    
    if (max_length == 0)
        max_length = 128;   // FIXME this should be a setting, or a formatting parameter
    
    clang_type_t clang_type = GetClangType();
    clang_type_t elem_or_pointee_clang_type;
    const Flags type_flags (ClangASTContext::GetTypeInfo (clang_type, 
                                                          GetClangAST(), 
                                                          &elem_or_pointee_clang_type));
    if (type_flags.AnySet (ClangASTContext::eTypeIsArray | ClangASTContext::eTypeIsPointer) &&
        ClangASTContext::IsCharType (elem_or_pointee_clang_type))
    {
        ExecutionContextScope *exe_scope = GetExecutionContextScope();
            if (exe_scope)
            {
                Target *target = exe_scope->CalculateTarget();
                if (target != NULL)
                {
                    lldb::addr_t cstr_address = LLDB_INVALID_ADDRESS;
                    AddressType cstr_address_type = eAddressTypeInvalid;
                    
                    size_t cstr_len = 0;
                    bool capped_data = false;
                    if (type_flags.Test (ClangASTContext::eTypeIsArray))
                    {
                        // We have an array
                        cstr_len = ClangASTContext::GetArraySize (clang_type);
                        if (cstr_len > max_length)
                        {
                            capped_data = true;
                            cstr_len = max_length;
                        }
                        cstr_address = GetAddressOf (cstr_address_type, true);
                    }
                    else
                    {
                        // We have a pointer
                        cstr_address = GetPointerValue (cstr_address_type, true);
                    }
                    if (cstr_address != LLDB_INVALID_ADDRESS)
                    {
                        Address cstr_so_addr (NULL, cstr_address);
                        DataExtractor data;
                        size_t bytes_read = 0;
                        std::vector<char> data_buffer;
                        bool prefer_file_cache = false;
                        if (cstr_len > 0 && honor_array)
                        {
                            data_buffer.resize(cstr_len);
                            data.SetData (&data_buffer.front(), data_buffer.size(), lldb::endian::InlHostByteOrder());
                            bytes_read = target->ReadMemory (cstr_so_addr, 
                                                             prefer_file_cache, 
                                                             &data_buffer.front(), 
                                                             cstr_len, 
                                                             error);
                            if (bytes_read > 0)
                            {
                                s << '"';
                                data.Dump (&s,
                                           0,                 // Start offset in "data"
                                           item_format,
                                           1,                 // Size of item (1 byte for a char!)
                                           bytes_read,        // How many bytes to print?
                                           UINT32_MAX,        // num per line
                                           LLDB_INVALID_ADDRESS,// base address
                                           0,                 // bitfield bit size
                                           0);                // bitfield bit offset
                                if (capped_data)
                                    s << "...";
                                s << '"';
                            }
                        }
                        else
                        {
                            cstr_len = max_length;
                            const size_t k_max_buf_size = 64;
                            data_buffer.resize (k_max_buf_size + 1);
                            // NULL terminate in case we don't get the entire C string
                            data_buffer.back() = '\0';
                            
                            s << '"';
                            
                            data.SetData (&data_buffer.front(), data_buffer.size(), endian::InlHostByteOrder());
                            while ((bytes_read = target->ReadMemory (cstr_so_addr, 
                                                                     prefer_file_cache,
                                                                     &data_buffer.front(), 
                                                                     k_max_buf_size, 
                                                                     error)) > 0)
                            {
                                size_t len = strlen(&data_buffer.front());
                                if (len == 0)
                                    break;
                                if (len > bytes_read)
                                    len = bytes_read;
                                if (len > cstr_len)
                                    len = cstr_len;
                                
                                data.Dump (&s,
                                           0,                 // Start offset in "data"
                                           item_format,
                                           1,                 // Size of item (1 byte for a char!)
                                           len,               // How many bytes to print?
                                           UINT32_MAX,        // num per line
                                           LLDB_INVALID_ADDRESS,// base address
                                           0,                 // bitfield bit size
                                           0);                // bitfield bit offset
                                
                                if (len < k_max_buf_size)
                                    break;
                                if (len >= cstr_len)
                                {
                                    s << "...";
                                    break;
                                }
                                cstr_len -= len;
                                cstr_so_addr.Slide (k_max_buf_size);
                            }
                            s << '"';
                        }
                    }
                }
            }
    }
    else
    {
        error.SetErrorString("impossible to read a string from this object");
    }
}

const char *
ValueObject::GetObjectDescription ()
{
    
    if (!UpdateValueIfNeeded ())
        return NULL;

    if (!m_object_desc_str.empty())
        return m_object_desc_str.c_str();

    ExecutionContextScope *exe_scope = GetExecutionContextScope();
    if (exe_scope == NULL)
        return NULL;
        
    Process *process = exe_scope->CalculateProcess();
    if (process == NULL)
        return NULL;
        
    StreamString s;
    
    lldb::LanguageType language = GetObjectRuntimeLanguage();
    LanguageRuntime *runtime = process->GetLanguageRuntime(language);
    
    if (runtime == NULL)
    {
        // Aw, hell, if the things a pointer, or even just an integer, let's try ObjC anyway...
        clang_type_t opaque_qual_type = GetClangType();
        if (opaque_qual_type != NULL)
        {
            bool is_signed;
            if (ClangASTContext::IsIntegerType (opaque_qual_type, is_signed) 
                || ClangASTContext::IsPointerType (opaque_qual_type))
            {
                runtime = process->GetLanguageRuntime(lldb::eLanguageTypeObjC);
            }
        }
    }
    
    if (runtime && runtime->GetObjectDescription(s, *this))
    {
        m_object_desc_str.append (s.GetData());
    }
    
    if (m_object_desc_str.empty())
        return NULL;
    else
        return m_object_desc_str.c_str();
}

const char *
ValueObject::GetValueAsCString ()
{
    // If our byte size is zero this is an aggregate type that has children
    if (ClangASTContext::IsAggregateType (GetClangType()) == false)
    {
        if (UpdateValueIfNeeded())
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
                            if (m_last_value_format)
                            {
                                m_value_str = m_last_value_format->FormatObject(GetSP());
                            }
                            else
                            {
                                StreamString sstr;
                                Format format = GetFormat();
                                if (format == eFormatDefault)                                
                                        format = (m_is_bitfield_for_scalar ? eFormatUnsigned :
                                        ClangASTType::GetFormat(clang_type));

                                if (ClangASTType::DumpTypeValue (GetClangAST(),            // The clang AST
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
                                {
                                    m_error.SetErrorStringWithFormat ("unsufficient data for value (only %u of %u bytes available)", 
                                                                      m_data.GetByteSize(),
                                                                      GetByteSize());
                                    m_value_str.clear();
                                }
                            }
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

// this call should only return pointers to data that needs no special memory management
// (either because they are hardcoded strings, or because they are backed by some other
// object); returning any new()-ed or malloc()-ed data here, will lead to leaks!
const char *
ValueObject::GetPrintableRepresentation(ValueObjectRepresentationStyle val_obj_display,
                                        lldb::Format custom_format)
{

    RefCounter ref(&m_dump_printable_counter);
    
    if (custom_format != lldb::eFormatInvalid)
        SetFormat(custom_format);
    
    const char * return_value;
    
    switch(val_obj_display)
    {
        case eDisplayValue:
            return_value = GetValueAsCString();
            break;
        case eDisplaySummary:
            return_value = GetSummaryAsCString();
            break;
        case eDisplayLanguageSpecific:
            return_value = GetObjectDescription();
            break;
        case eDisplayLocation:
            return_value = GetLocationAsCString();
            break;
    }
    
    // this code snippet might lead to endless recursion, thus we use a RefCounter here to
    // check that we are not looping endlessly
    if (!return_value && (m_dump_printable_counter < 3))
    {
        // try to pick the other choice
        if (val_obj_display == eDisplayValue)
            return_value = GetSummaryAsCString();
        else if (val_obj_display == eDisplaySummary)
        {
            if (ClangASTContext::IsAggregateType (GetClangType()) == true)
            {
                // this thing has no value
                return_value = "<no summary defined for this datatype>";
            }
            else
                return_value = GetValueAsCString();
        }
    }
    
    return (return_value ? return_value : "<no printable representation>");

}

bool
ValueObject::DumpPrintableRepresentation(Stream& s,
                                         ValueObjectRepresentationStyle val_obj_display,
                                         lldb::Format custom_format)
{

    clang_type_t elem_or_pointee_type;
    Flags flags(ClangASTContext::GetTypeInfo(GetClangType(), GetClangAST(), &elem_or_pointee_type));
    
    if (flags.AnySet(ClangASTContext::eTypeIsArray | ClangASTContext::eTypeIsPointer)
         && val_obj_display == ValueObject::eDisplayValue)
    {
        // when being asked to get a printable display an array or pointer type directly, 
        // try to "do the right thing"
        
        if (IsCStringContainer(true) && 
            (custom_format == lldb::eFormatCString ||
             custom_format == lldb::eFormatCharArray ||
             custom_format == lldb::eFormatChar ||
             custom_format == lldb::eFormatVectorOfChar)) // print char[] & char* directly
        {
            Error error;
            ReadPointedString(s,
                              error,
                              0,
                              (custom_format == lldb::eFormatVectorOfChar) ||
                              (custom_format == lldb::eFormatCharArray));
            return !error.Fail();
        }
        
        if (custom_format == lldb::eFormatEnum)
            return false;
        
        // this only works for arrays, because I have no way to know when
        // the pointed memory ends, and no special \0 end of data marker
        if (flags.Test(ClangASTContext::eTypeIsArray))
        {
            if ((custom_format == lldb::eFormatBytes) ||
                (custom_format == lldb::eFormatBytesWithASCII))
            {
                uint32_t count = GetNumChildren();
                                
                s << '[';
                for (uint32_t low = 0; low < count; low++)
                {
                    
                    if (low)
                        s << ',';
                    
                    ValueObjectSP child = GetChildAtIndex(low,true);
                    if (!child.get())
                    {
                        s << "<invalid child>";
                        continue;
                    }
                    child->DumpPrintableRepresentation(s, ValueObject::eDisplayValue, custom_format);
                }                
                
                s << ']';
                
                return true;
            }
            
            if ((custom_format == lldb::eFormatVectorOfChar) ||
                (custom_format == lldb::eFormatVectorOfFloat32) ||
                (custom_format == lldb::eFormatVectorOfFloat64) ||
                (custom_format == lldb::eFormatVectorOfSInt16) ||
                (custom_format == lldb::eFormatVectorOfSInt32) ||
                (custom_format == lldb::eFormatVectorOfSInt64) ||
                (custom_format == lldb::eFormatVectorOfSInt8) ||
                (custom_format == lldb::eFormatVectorOfUInt128) ||
                (custom_format == lldb::eFormatVectorOfUInt16) ||
                (custom_format == lldb::eFormatVectorOfUInt32) ||
                (custom_format == lldb::eFormatVectorOfUInt64) ||
                (custom_format == lldb::eFormatVectorOfUInt8)) // arrays of bytes, bytes with ASCII or any vector format should be printed directly
            {
                uint32_t count = GetNumChildren();

                lldb::Format format = FormatManager::GetSingleItemFormat(custom_format);
                
                s << '[';
                for (uint32_t low = 0; low < count; low++)
                {
                    
                    if (low)
                        s << ',';
                    
                    ValueObjectSP child = GetChildAtIndex(low,true);
                    if (!child.get())
                    {
                        s << "<invalid child>";
                        continue;
                    }
                    child->DumpPrintableRepresentation(s, ValueObject::eDisplayValue, format);
                }                
                
                s << ']';
                
                return true;
            }
        }
        
        if ((custom_format == lldb::eFormatBoolean) ||
            (custom_format == lldb::eFormatBinary) ||
            (custom_format == lldb::eFormatChar) ||
            (custom_format == lldb::eFormatCharPrintable) ||
            (custom_format == lldb::eFormatComplexFloat) ||
            (custom_format == lldb::eFormatDecimal) ||
            (custom_format == lldb::eFormatHex) ||
            (custom_format == lldb::eFormatFloat) ||
            (custom_format == lldb::eFormatOctal) ||
            (custom_format == lldb::eFormatOSType) ||
            (custom_format == lldb::eFormatUnicode16) ||
            (custom_format == lldb::eFormatUnicode32) ||
            (custom_format == lldb::eFormatUnsigned) ||
            (custom_format == lldb::eFormatPointer) ||
            (custom_format == lldb::eFormatComplexInteger) ||
            (custom_format == lldb::eFormatComplex) ||
            (custom_format == lldb::eFormatDefault)) // use the [] operator
            return false;
    }
    const char *targetvalue = GetPrintableRepresentation(val_obj_display, custom_format);
    if (targetvalue)
        s.PutCString(targetvalue);
    bool var_success = (targetvalue != NULL);
    if (custom_format != eFormatInvalid)
        SetFormat(eFormatDefault);
    return var_success;
}

addr_t
ValueObject::GetAddressOf (AddressType &address_type, bool scalar_is_load_address)
{
    if (!UpdateValueIfNeeded())
        return LLDB_INVALID_ADDRESS;
        
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
ValueObject::GetPointerValue (AddressType &address_type, bool scalar_is_load_address)
{
    lldb::addr_t address = LLDB_INVALID_ADDRESS;
    address_type = eAddressTypeInvalid;
    
    if (!UpdateValueIfNeeded())
        return address;
        
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
ValueObject::SetValueFromCString (const char *value_str)
{
    // Make sure our value is up to date first so that our location and location
    // type is valid.
    if (!UpdateValueIfNeeded())
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
            m_value.GetScalar() = ull_val;
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
            m_value.GetScalar() = sll_val;
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
                m_data.SetByteOrder(lldb::endian::InlHostByteOrder());
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
    m_update_point.SetNeedsUpdate();

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
ValueObject::AddSyntheticChild (const ConstString &key, ValueObject *valobj)
{
    m_synthetic_children[key] = valobj;
}

ValueObjectSP
ValueObject::GetSyntheticChild (const ConstString &key) const
{
    ValueObjectSP synthetic_child_sp;
    std::map<ConstString, ValueObject *>::const_iterator pos = m_synthetic_children.find (key);
    if (pos != m_synthetic_children.end())
        synthetic_child_sp = pos->second->GetSP();
    return synthetic_child_sp;
}

bool
ValueObject::IsPointerType ()
{
    return ClangASTContext::IsPointerType (GetClangType());
}

bool
ValueObject::IsArrayType ()
{
    return ClangASTContext::IsArrayType (GetClangType());
}

bool
ValueObject::IsScalarType ()
{
    return ClangASTContext::IsScalarType (GetClangType());
}

bool
ValueObject::IsIntegerType (bool &is_signed)
{
    return ClangASTContext::IsIntegerType (GetClangType(), is_signed);
}

bool
ValueObject::IsPointerOrReferenceType ()
{
    return ClangASTContext::IsPointerOrReferenceType (GetClangType());
}

bool
ValueObject::IsPossibleCPlusPlusDynamicType ()
{
    return ClangASTContext::IsPossibleCPlusPlusDynamicType (GetClangAST (), GetClangType());
}

bool
ValueObject::IsPossibleDynamicType ()
{
    return ClangASTContext::IsPossibleDynamicType (GetClangAST (), GetClangType());
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
            ValueObject *synthetic_child;
            // We haven't made a synthetic array member for INDEX yet, so
            // lets make one and cache it for any future reference.
            synthetic_child = CreateChildAtIndex(0, true, index);

            // Cache the value if we got one back...
            if (synthetic_child)
            {
                AddSyntheticChild(index_const_str, synthetic_child);
                synthetic_child_sp = synthetic_child->GetSP();
                synthetic_child_sp->SetName(index_str);
                synthetic_child_sp->m_is_array_item_for_pointer = true;
            }
        }
    }
    return synthetic_child_sp;
}

// This allows you to create an array member using and index
// that doesn't not fall in the normal bounds of the array.
// Many times structure can be defined as:
// struct Collection
// {
//     uint32_t item_count;
//     Item item_array[0];
// };
// The size of the "item_array" is 1, but many times in practice
// there are more items in "item_array".

ValueObjectSP
ValueObject::GetSyntheticArrayMemberFromArray (int32_t index, bool can_create)
{
    ValueObjectSP synthetic_child_sp;
    if (IsArrayType ())
    {
        char index_str[64];
        snprintf(index_str, sizeof(index_str), "[%i]", index);
        ConstString index_const_str(index_str);
        // Check if we have already created a synthetic array member in this
        // valid object. If we have we will re-use it.
        synthetic_child_sp = GetSyntheticChild (index_const_str);
        if (!synthetic_child_sp)
        {
            ValueObject *synthetic_child;
            // We haven't made a synthetic array member for INDEX yet, so
            // lets make one and cache it for any future reference.
            synthetic_child = CreateChildAtIndex(0, true, index);
            
            // Cache the value if we got one back...
            if (synthetic_child)
            {
                AddSyntheticChild(index_const_str, synthetic_child);
                synthetic_child_sp = synthetic_child->GetSP();
                synthetic_child_sp->SetName(index_str);
                synthetic_child_sp->m_is_array_item_for_pointer = true;
            }
        }
    }
    return synthetic_child_sp;
}

ValueObjectSP
ValueObject::GetSyntheticBitFieldChild (uint32_t from, uint32_t to, bool can_create)
{
    ValueObjectSP synthetic_child_sp;
    if (IsScalarType ())
    {
        char index_str[64];
        snprintf(index_str, sizeof(index_str), "[%i-%i]", from, to);
        ConstString index_const_str(index_str);
        // Check if we have already created a synthetic array member in this
        // valid object. If we have we will re-use it.
        synthetic_child_sp = GetSyntheticChild (index_const_str);
        if (!synthetic_child_sp)
        {
            ValueObjectChild *synthetic_child;
            // We haven't made a synthetic array member for INDEX yet, so
            // lets make one and cache it for any future reference.
            synthetic_child = new ValueObjectChild(*this,
                                                      GetClangAST(),
                                                      GetClangType(),
                                                      index_const_str,
                                                      GetByteSize(),
                                                      0,
                                                      to-from+1,
                                                      from,
                                                      false,
                                                      false);
            
            // Cache the value if we got one back...
            if (synthetic_child)
            {
                AddSyntheticChild(index_const_str, synthetic_child);
                synthetic_child_sp = synthetic_child->GetSP();
                synthetic_child_sp->SetName(index_str);
                synthetic_child_sp->m_is_bitfield_for_scalar = true;
            }
        }
    }
    return synthetic_child_sp;
}

// your expression path needs to have a leading . or ->
// (unless it somehow "looks like" an array, in which case it has
// a leading [ symbol). while the [ is meaningful and should be shown
// to the user, . and -> are just parser design, but by no means
// added information for the user.. strip them off
static const char*
SkipLeadingExpressionPathSeparators(const char* expression)
{
    if (!expression || !expression[0])
        return expression;
    if (expression[0] == '.')
        return expression+1;
    if (expression[0] == '-' && expression[1] == '>')
        return expression+2;
    return expression;
}

lldb::ValueObjectSP
ValueObject::GetSyntheticExpressionPathChild(const char* expression, bool can_create)
{
    ValueObjectSP synthetic_child_sp;
    ConstString name_const_string(expression);
    // Check if we have already created a synthetic array member in this
    // valid object. If we have we will re-use it.
    synthetic_child_sp = GetSyntheticChild (name_const_string);
    if (!synthetic_child_sp)
    {
        // We haven't made a synthetic array member for expression yet, so
        // lets make one and cache it for any future reference.
        synthetic_child_sp = GetValueForExpressionPath(expression);
        
        // Cache the value if we got one back...
        if (synthetic_child_sp.get())
        {
            AddSyntheticChild(name_const_string, synthetic_child_sp.get());
            synthetic_child_sp->SetName(SkipLeadingExpressionPathSeparators(expression));
            synthetic_child_sp->m_is_expression_path_child = true;
        }
    }
    return synthetic_child_sp;
}

void
ValueObject::CalculateSyntheticValue (lldb::SyntheticValueType use_synthetic)
{
    if (use_synthetic == lldb::eNoSyntheticFilter)
        return;
    
    UpdateFormatsIfNeeded();
    
    if (m_last_synthetic_filter.get() == NULL)
        return;
    
    if (m_synthetic_value == NULL)
        m_synthetic_value = new ValueObjectSynthetic(*this, m_last_synthetic_filter);
    
}

void
ValueObject::CalculateDynamicValue (lldb::DynamicValueType use_dynamic)
{
    if (use_dynamic == lldb::eNoDynamicValues)
        return;
        
    if (!m_dynamic_value && !IsDynamic())
    {
        Process *process = m_update_point.GetProcess();
        bool worth_having_dynamic_value = false;
        
        
        // FIXME: Process should have some kind of "map over Runtimes" so we don't have to
        // hard code this everywhere.
        lldb::LanguageType known_type = GetObjectRuntimeLanguage();
        if (known_type != lldb::eLanguageTypeUnknown && known_type != lldb::eLanguageTypeC)
        {
            LanguageRuntime *runtime = process->GetLanguageRuntime (known_type);
            if (runtime)
                worth_having_dynamic_value = runtime->CouldHaveDynamicValue(*this);
        }
        else
        {
            LanguageRuntime *cpp_runtime = process->GetLanguageRuntime (lldb::eLanguageTypeC_plus_plus);
            if (cpp_runtime)
                worth_having_dynamic_value = cpp_runtime->CouldHaveDynamicValue(*this);
            
            if (!worth_having_dynamic_value)
            {
                LanguageRuntime *objc_runtime = process->GetLanguageRuntime (lldb::eLanguageTypeObjC);
                if (objc_runtime)
                    worth_having_dynamic_value = objc_runtime->CouldHaveDynamicValue(*this);
            }
        }
        
        if (worth_having_dynamic_value)
            m_dynamic_value = new ValueObjectDynamicValue (*this, use_dynamic);
            
//        if (worth_having_dynamic_value)
//            printf ("Adding dynamic value %s (%p) to (%p) - manager %p.\n", m_name.GetCString(), m_dynamic_value, this, m_manager);

    }
}

ValueObjectSP
ValueObject::GetDynamicValue (DynamicValueType use_dynamic)
{
    if (use_dynamic == lldb::eNoDynamicValues)
        return ValueObjectSP();
        
    if (!IsDynamic() && m_dynamic_value == NULL)
    {
        CalculateDynamicValue(use_dynamic);
    }
    if (m_dynamic_value)
        return m_dynamic_value->GetSP();
    else
        return ValueObjectSP();
}

// GetDynamicValue() returns a NULL SharedPointer if the object is not dynamic
// or we do not really want a dynamic VO. this method instead returns this object
// itself when making it synthetic has no meaning. this makes it much simpler
// to replace the SyntheticValue for the ValueObject
ValueObjectSP
ValueObject::GetSyntheticValue (SyntheticValueType use_synthetic)
{
    if (use_synthetic == lldb::eNoSyntheticFilter)
        return GetSP();
    
    UpdateFormatsIfNeeded();
    
    if (m_last_synthetic_filter.get() == NULL)
        return GetSP();
    
    CalculateSyntheticValue(use_synthetic);
    
    if (m_synthetic_value)
        return m_synthetic_value->GetSP();
    else
        return GetSP();
}

bool
ValueObject::GetBaseClassPath (Stream &s)
{
    if (IsBaseClass())
    {
        bool parent_had_base_class = GetParent() && GetParent()->GetBaseClassPath (s);
        clang_type_t clang_type = GetClangType();
        std::string cxx_class_name;
        bool this_had_base_class = ClangASTContext::GetCXXClassName (clang_type, cxx_class_name);
        if (this_had_base_class)
        {
            if (parent_had_base_class)
                s.PutCString("::");
            s.PutCString(cxx_class_name.c_str());
        }
        return parent_had_base_class || this_had_base_class;
    }
    return false;
}


ValueObject *
ValueObject::GetNonBaseClassParent()
{
    if (GetParent())
    {
        if (GetParent()->IsBaseClass())
            return GetParent()->GetNonBaseClassParent();
        else
            return GetParent();
    }
    return NULL;
}

void
ValueObject::GetExpressionPath (Stream &s, bool qualify_cxx_base_classes, GetExpressionPathFormat epformat)
{
    const bool is_deref_of_parent = IsDereferenceOfParent ();

    if (is_deref_of_parent && epformat == eDereferencePointers) {
        // this is the original format of GetExpressionPath() producing code like *(a_ptr).memberName, which is entirely
        // fine, until you put this into StackFrame::GetValueForVariableExpressionPath() which prefers to see a_ptr->memberName.
        // the eHonorPointers mode is meant to produce strings in this latter format
        s.PutCString("*(");
    }
    
    ValueObject* parent = GetParent();
    
    if (parent)
        parent->GetExpressionPath (s, qualify_cxx_base_classes, epformat);
    
    // if we are a deref_of_parent just because we are synthetic array
    // members made up to allow ptr[%d] syntax to work in variable
    // printing, then add our name ([%d]) to the expression path
    if (m_is_array_item_for_pointer && epformat == eHonorPointers)
        s.PutCString(m_name.AsCString());
            
    if (!IsBaseClass())
    {
        if (!is_deref_of_parent)
        {
            ValueObject *non_base_class_parent = GetNonBaseClassParent();
            if (non_base_class_parent)
            {
                clang_type_t non_base_class_parent_clang_type = non_base_class_parent->GetClangType();
                if (non_base_class_parent_clang_type)
                {
                    const uint32_t non_base_class_parent_type_info = ClangASTContext::GetTypeInfo (non_base_class_parent_clang_type, NULL, NULL);
                    
                    if (parent && parent->IsDereferenceOfParent() && epformat == eHonorPointers)
                    {
                        s.PutCString("->");
                    }
                    else
                    {                    
                        if (non_base_class_parent_type_info & ClangASTContext::eTypeIsPointer)
                        {
                            s.PutCString("->");
                        }
                        else if ((non_base_class_parent_type_info & ClangASTContext::eTypeHasChildren) &&
                                 !(non_base_class_parent_type_info & ClangASTContext::eTypeIsArray))
                        {
                            s.PutChar('.');
                        }
                    }
                }
            }

            const char *name = GetName().GetCString();
            if (name)
            {
                if (qualify_cxx_base_classes)
                {
                    if (GetBaseClassPath (s))
                        s.PutCString("::");
                }
                s.PutCString(name);
            }
        }
    }
    
    if (is_deref_of_parent && epformat == eDereferencePointers) {
        s.PutChar(')');
    }
}

lldb::ValueObjectSP
ValueObject::GetValueForExpressionPath(const char* expression,
                                       const char** first_unparsed,
                                       ExpressionPathScanEndReason* reason_to_stop,
                                       ExpressionPathEndResultType* final_value_type,
                                       const GetValueForExpressionPathOptions& options,
                                       ExpressionPathAftermath* final_task_on_target)
{
    
    const char* dummy_first_unparsed;
    ExpressionPathScanEndReason dummy_reason_to_stop;
    ExpressionPathEndResultType dummy_final_value_type;
    ExpressionPathAftermath dummy_final_task_on_target = ValueObject::eNothing;
    
    ValueObjectSP ret_val = GetValueForExpressionPath_Impl(expression,
                                                           first_unparsed ? first_unparsed : &dummy_first_unparsed,
                                                           reason_to_stop ? reason_to_stop : &dummy_reason_to_stop,
                                                           final_value_type ? final_value_type : &dummy_final_value_type,
                                                           options,
                                                           final_task_on_target ? final_task_on_target : &dummy_final_task_on_target);
    
    if (!final_task_on_target || *final_task_on_target == ValueObject::eNothing)
    {
        return ret_val;
    }
    if (ret_val.get() && *final_value_type == ePlain) // I can only deref and takeaddress of plain objects
    {
        if (*final_task_on_target == ValueObject::eDereference)
        {
            Error error;
            ValueObjectSP final_value = ret_val->Dereference(error);
            if (error.Fail() || !final_value.get())
            {
                *reason_to_stop = ValueObject::eDereferencingFailed;
                *final_value_type = ValueObject::eInvalid;
                return ValueObjectSP();
            }
            else
            {
                *final_task_on_target = ValueObject::eNothing;
                return final_value;
            }
        }
        if (*final_task_on_target == ValueObject::eTakeAddress)
        {
            Error error;
            ValueObjectSP final_value = ret_val->AddressOf(error);
            if (error.Fail() || !final_value.get())
            {
                *reason_to_stop = ValueObject::eTakingAddressFailed;
                *final_value_type = ValueObject::eInvalid;
                return ValueObjectSP();
            }
            else
            {
                *final_task_on_target = ValueObject::eNothing;
                return final_value;
            }
        }
    }
    return ret_val; // final_task_on_target will still have its original value, so you know I did not do it
}

int
ValueObject::GetValuesForExpressionPath(const char* expression,
                                        lldb::ValueObjectListSP& list,
                                        const char** first_unparsed,
                                        ExpressionPathScanEndReason* reason_to_stop,
                                        ExpressionPathEndResultType* final_value_type,
                                        const GetValueForExpressionPathOptions& options,
                                        ExpressionPathAftermath* final_task_on_target)
{
    const char* dummy_first_unparsed;
    ExpressionPathScanEndReason dummy_reason_to_stop;
    ExpressionPathEndResultType dummy_final_value_type;
    ExpressionPathAftermath dummy_final_task_on_target = ValueObject::eNothing;
    
    ValueObjectSP ret_val = GetValueForExpressionPath_Impl(expression,
                                                           first_unparsed ? first_unparsed : &dummy_first_unparsed,
                                                           reason_to_stop ? reason_to_stop : &dummy_reason_to_stop,
                                                           final_value_type ? final_value_type : &dummy_final_value_type,
                                                           options,
                                                           final_task_on_target ? final_task_on_target : &dummy_final_task_on_target);
    
    if (!ret_val.get()) // if there are errors, I add nothing to the list
        return 0;
    
    if (*reason_to_stop != eArrayRangeOperatorMet)
    {
        // I need not expand a range, just post-process the final value and return
        if (!final_task_on_target || *final_task_on_target == ValueObject::eNothing)
        {
            list->Append(ret_val);
            return 1;
        }
        if (ret_val.get() && *final_value_type == ePlain) // I can only deref and takeaddress of plain objects
        {
            if (*final_task_on_target == ValueObject::eDereference)
            {
                Error error;
                ValueObjectSP final_value = ret_val->Dereference(error);
                if (error.Fail() || !final_value.get())
                {
                    *reason_to_stop = ValueObject::eDereferencingFailed;
                    *final_value_type = ValueObject::eInvalid;
                    return 0;
                }
                else
                {
                    *final_task_on_target = ValueObject::eNothing;
                    list->Append(final_value);
                    return 1;
                }
            }
            if (*final_task_on_target == ValueObject::eTakeAddress)
            {
                Error error;
                ValueObjectSP final_value = ret_val->AddressOf(error);
                if (error.Fail() || !final_value.get())
                {
                    *reason_to_stop = ValueObject::eTakingAddressFailed;
                    *final_value_type = ValueObject::eInvalid;
                    return 0;
                }
                else
                {
                    *final_task_on_target = ValueObject::eNothing;
                    list->Append(final_value);
                    return 1;
                }
            }
        }
    }
    else
    {
        return ExpandArraySliceExpression(first_unparsed ? *first_unparsed : dummy_first_unparsed,
                                          first_unparsed ? first_unparsed : &dummy_first_unparsed,
                                          ret_val,
                                          list,
                                          reason_to_stop ? reason_to_stop : &dummy_reason_to_stop,
                                          final_value_type ? final_value_type : &dummy_final_value_type,
                                          options,
                                          final_task_on_target ? final_task_on_target : &dummy_final_task_on_target);
    }
    // in any non-covered case, just do the obviously right thing
    list->Append(ret_val);
    return 1;
}

lldb::ValueObjectSP
ValueObject::GetValueForExpressionPath_Impl(const char* expression_cstr,
                                            const char** first_unparsed,
                                            ExpressionPathScanEndReason* reason_to_stop,
                                            ExpressionPathEndResultType* final_result,
                                            const GetValueForExpressionPathOptions& options,
                                            ExpressionPathAftermath* what_next)
{
    ValueObjectSP root = GetSP();
    
    if (!root.get())
        return ValueObjectSP();
    
    *first_unparsed = expression_cstr;
    
    while (true)
    {
        
        const char* expression_cstr = *first_unparsed; // hide the top level expression_cstr
        
        lldb::clang_type_t root_clang_type = root->GetClangType();
        lldb::clang_type_t pointee_clang_type;
        Flags root_clang_type_info,pointee_clang_type_info;
        
        root_clang_type_info = Flags(ClangASTContext::GetTypeInfo(root_clang_type, GetClangAST(), &pointee_clang_type));
        if (pointee_clang_type)
            pointee_clang_type_info = Flags(ClangASTContext::GetTypeInfo(pointee_clang_type, GetClangAST(), NULL));
        
        if (!expression_cstr || *expression_cstr == '\0')
        {
            *reason_to_stop = ValueObject::eEndOfString;
            return root;
        }
        
        switch (*expression_cstr)
        {
            case '-':
            {
                if (options.m_check_dot_vs_arrow_syntax &&
                    root_clang_type_info.Test(ClangASTContext::eTypeIsPointer) ) // if you are trying to use -> on a non-pointer and I must catch the error
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eArrowInsteadOfDot;
                    *final_result = ValueObject::eInvalid;
                    return ValueObjectSP();
                }
                if (root_clang_type_info.Test(ClangASTContext::eTypeIsObjC) &&  // if yo are trying to extract an ObjC IVar when this is forbidden
                    root_clang_type_info.Test(ClangASTContext::eTypeIsPointer) &&
                    options.m_no_fragile_ivar)
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eFragileIVarNotAllowed;
                    *final_result = ValueObject::eInvalid;
                    return ValueObjectSP();
                }
                if (expression_cstr[1] != '>')
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eUnexpectedSymbol;
                    *final_result = ValueObject::eInvalid;
                    return ValueObjectSP();
                }
                expression_cstr++; // skip the -
            }
            case '.': // or fallthrough from ->
            {
                if (options.m_check_dot_vs_arrow_syntax && *expression_cstr == '.' &&
                    root_clang_type_info.Test(ClangASTContext::eTypeIsPointer)) // if you are trying to use . on a pointer and I must catch the error
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eDotInsteadOfArrow;
                    *final_result = ValueObject::eInvalid;
                    return ValueObjectSP();
                }
                expression_cstr++; // skip .
                const char *next_separator = strpbrk(expression_cstr+1,"-.[");
                ConstString child_name;
                if (!next_separator) // if no other separator just expand this last layer
                {
                    child_name.SetCString (expression_cstr);
                    root = root->GetChildMemberWithName(child_name, true);
                    if (root.get()) // we know we are done, so just return
                    {
                        *first_unparsed = '\0';
                        *reason_to_stop = ValueObject::eEndOfString;
                        *final_result = ValueObject::ePlain;
                        return root;
                    }
                    else
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eNoSuchChild;
                        *final_result = ValueObject::eInvalid;
                        return ValueObjectSP();
                    }
                }
                else // other layers do expand
                {
                    child_name.SetCStringWithLength(expression_cstr, next_separator - expression_cstr);
                    root = root->GetChildMemberWithName(child_name, true);
                    if (root.get()) // store the new root and move on
                    {
                        *first_unparsed = next_separator;
                        *final_result = ValueObject::ePlain;
                        continue;
                    }
                    else
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eNoSuchChild;
                        *final_result = ValueObject::eInvalid;
                        return ValueObjectSP();
                    }
                }
                break;
            }
            case '[':
            {
                if (!root_clang_type_info.Test(ClangASTContext::eTypeIsArray) && !root_clang_type_info.Test(ClangASTContext::eTypeIsPointer)) // if this is not a T[] nor a T*
                {
                    if (!root_clang_type_info.Test(ClangASTContext::eTypeIsScalar)) // if this is not even a scalar, this syntax is just plain wrong!
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eRangeOperatorInvalid;
                        *final_result = ValueObject::eInvalid;
                        return ValueObjectSP();
                    }
                    else if (!options.m_allow_bitfields_syntax) // if this is a scalar, check that we can expand bitfields
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eRangeOperatorNotAllowed;
                        *final_result = ValueObject::eInvalid;
                        return ValueObjectSP();
                    }
                }
                if (*(expression_cstr+1) == ']') // if this is an unbounded range it only works for arrays
                {
                    if (!root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eEmptyRangeNotAllowed;
                        *final_result = ValueObject::eInvalid;
                        return ValueObjectSP();
                    }
                    else // even if something follows, we cannot expand unbounded ranges, just let the caller do it
                    {
                        *first_unparsed = expression_cstr+2;
                        *reason_to_stop = ValueObject::eArrayRangeOperatorMet;
                        *final_result = ValueObject::eUnboundedRange;
                        return root;
                    }
                }
                const char *separator_position = ::strchr(expression_cstr+1,'-');
                const char *close_bracket_position = ::strchr(expression_cstr+1,']');
                if (!close_bracket_position) // if there is no ], this is a syntax error
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eUnexpectedSymbol;
                    *final_result = ValueObject::eInvalid;
                    return ValueObjectSP();
                }
                if (!separator_position || separator_position > close_bracket_position) // if no separator, this is either [] or [N]
                {
                    char *end = NULL;
                    unsigned long index = ::strtoul (expression_cstr+1, &end, 0);
                    if (!end || end != close_bracket_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eUnexpectedSymbol;
                        *final_result = ValueObject::eInvalid;
                        return ValueObjectSP();
                    }
                    if (end - expression_cstr == 1) // if this is [], only return a valid value for arrays
                    {
                        if (root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                        {
                            *first_unparsed = expression_cstr+2;
                            *reason_to_stop = ValueObject::eArrayRangeOperatorMet;
                            *final_result = ValueObject::eUnboundedRange;
                            return root;
                        }
                        else
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eEmptyRangeNotAllowed;
                            *final_result = ValueObject::eInvalid;
                            return ValueObjectSP();
                        }
                    }
                    // from here on we do have a valid index
                    if (root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                    {
                        ValueObjectSP child_valobj_sp = root->GetChildAtIndex(index, true);
                        if (!child_valobj_sp)
                            child_valobj_sp = root->GetSyntheticArrayMemberFromArray(index, true);
                        if (child_valobj_sp)
                        {
                            root = child_valobj_sp;
                            *first_unparsed = end+1; // skip ]
                            *final_result = ValueObject::ePlain;
                            continue;
                        }
                        else
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eNoSuchChild;
                            *final_result = ValueObject::eInvalid;
                            return ValueObjectSP();
                        }
                    }
                    else if (root_clang_type_info.Test(ClangASTContext::eTypeIsPointer))
                    {
                        if (*what_next == ValueObject::eDereference &&  // if this is a ptr-to-scalar, I am accessing it by index and I would have deref'ed anyway, then do it now and use this as a bitfield
                            pointee_clang_type_info.Test(ClangASTContext::eTypeIsScalar))
                        {
                            Error error;
                            root = root->Dereference(error);
                            if (error.Fail() || !root.get())
                            {
                                *first_unparsed = expression_cstr;
                                *reason_to_stop = ValueObject::eDereferencingFailed;
                                *final_result = ValueObject::eInvalid;
                                return ValueObjectSP();
                            }
                            else
                            {
                                *what_next = eNothing;
                                continue;
                            }
                        }
                        else
                        {
                            root = root->GetSyntheticArrayMemberFromPointer(index, true);
                            if (!root.get())
                            {
                                *first_unparsed = expression_cstr;
                                *reason_to_stop = ValueObject::eNoSuchChild;
                                *final_result = ValueObject::eInvalid;
                                return ValueObjectSP();
                            }
                            else
                            {
                                *first_unparsed = end+1; // skip ]
                                *final_result = ValueObject::ePlain;
                                continue;
                            }
                        }
                    }
                    else /*if (ClangASTContext::IsScalarType(root_clang_type))*/
                    {
                        root = root->GetSyntheticBitFieldChild(index, index, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eNoSuchChild;
                            *final_result = ValueObject::eInvalid;
                            return ValueObjectSP();
                        }
                        else // we do not know how to expand members of bitfields, so we just return and let the caller do any further processing
                        {
                            *first_unparsed = end+1; // skip ]
                            *reason_to_stop = ValueObject::eBitfieldRangeOperatorMet;
                            *final_result = ValueObject::eBitfield;
                            return root;
                        }
                    }
                }
                else // we have a low and a high index
                {
                    char *end = NULL;
                    unsigned long index_lower = ::strtoul (expression_cstr+1, &end, 0);
                    if (!end || end != separator_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eUnexpectedSymbol;
                        *final_result = ValueObject::eInvalid;
                        return ValueObjectSP();
                    }
                    unsigned long index_higher = ::strtoul (separator_position+1, &end, 0);
                    if (!end || end != close_bracket_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eUnexpectedSymbol;
                        *final_result = ValueObject::eInvalid;
                        return ValueObjectSP();
                    }
                    if (index_lower > index_higher) // swap indices if required
                    {
                        unsigned long temp = index_lower;
                        index_lower = index_higher;
                        index_higher = temp;
                    }
                    if (root_clang_type_info.Test(ClangASTContext::eTypeIsScalar)) // expansion only works for scalars
                    {
                        root = root->GetSyntheticBitFieldChild(index_lower, index_higher, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eNoSuchChild;
                            *final_result = ValueObject::eInvalid;
                            return ValueObjectSP();
                        }
                        else
                        {
                            *first_unparsed = end+1; // skip ]
                            *reason_to_stop = ValueObject::eBitfieldRangeOperatorMet;
                            *final_result = ValueObject::eBitfield;
                            return root;
                        }
                    }
                    else if (root_clang_type_info.Test(ClangASTContext::eTypeIsPointer) && // if this is a ptr-to-scalar, I am accessing it by index and I would have deref'ed anyway, then do it now and use this as a bitfield
                             *what_next == ValueObject::eDereference &&
                             pointee_clang_type_info.Test(ClangASTContext::eTypeIsScalar))
                    {
                        Error error;
                        root = root->Dereference(error);
                        if (error.Fail() || !root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eDereferencingFailed;
                            *final_result = ValueObject::eInvalid;
                            return ValueObjectSP();
                        }
                        else
                        {
                            *what_next = ValueObject::eNothing;
                            continue;
                        }
                    }
                    else
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eArrayRangeOperatorMet;
                        *final_result = ValueObject::eBoundedRange;
                        return root;
                    }
                }
                break;
            }
            default: // some non-separator is in the way
            {
                *first_unparsed = expression_cstr;
                *reason_to_stop = ValueObject::eUnexpectedSymbol;
                *final_result = ValueObject::eInvalid;
                return ValueObjectSP();
                break;
            }
        }
    }
}

int
ValueObject::ExpandArraySliceExpression(const char* expression_cstr,
                                        const char** first_unparsed,
                                        lldb::ValueObjectSP root,
                                        lldb::ValueObjectListSP& list,
                                        ExpressionPathScanEndReason* reason_to_stop,
                                        ExpressionPathEndResultType* final_result,
                                        const GetValueForExpressionPathOptions& options,
                                        ExpressionPathAftermath* what_next)
{
    if (!root.get())
        return 0;
    
    *first_unparsed = expression_cstr;
    
    while (true)
    {
        
        const char* expression_cstr = *first_unparsed; // hide the top level expression_cstr
        
        lldb::clang_type_t root_clang_type = root->GetClangType();
        lldb::clang_type_t pointee_clang_type;
        Flags root_clang_type_info,pointee_clang_type_info;
        
        root_clang_type_info = Flags(ClangASTContext::GetTypeInfo(root_clang_type, GetClangAST(), &pointee_clang_type));
        if (pointee_clang_type)
            pointee_clang_type_info = Flags(ClangASTContext::GetTypeInfo(pointee_clang_type, GetClangAST(), NULL));
        
        if (!expression_cstr || *expression_cstr == '\0')
        {
            *reason_to_stop = ValueObject::eEndOfString;
            list->Append(root);
            return 1;
        }
        
        switch (*expression_cstr)
        {
            case '[':
            {
                if (!root_clang_type_info.Test(ClangASTContext::eTypeIsArray) && !root_clang_type_info.Test(ClangASTContext::eTypeIsPointer)) // if this is not a T[] nor a T*
                {
                    if (!root_clang_type_info.Test(ClangASTContext::eTypeIsScalar)) // if this is not even a scalar, this syntax is just plain wrong!
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eRangeOperatorInvalid;
                        *final_result = ValueObject::eInvalid;
                        return 0;
                    }
                    else if (!options.m_allow_bitfields_syntax) // if this is a scalar, check that we can expand bitfields
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eRangeOperatorNotAllowed;
                        *final_result = ValueObject::eInvalid;
                        return 0;
                    }
                }
                if (*(expression_cstr+1) == ']') // if this is an unbounded range it only works for arrays
                {
                    if (!root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eEmptyRangeNotAllowed;
                        *final_result = ValueObject::eInvalid;
                        return 0;
                    }
                    else // expand this into list
                    {
                        int max_index = root->GetNumChildren() - 1;
                        for (int index = 0; index < max_index; index++)
                        {
                            ValueObjectSP child = 
                                root->GetChildAtIndex(index, true);
                            list->Append(child);
                        }
                        *first_unparsed = expression_cstr+2;
                        *reason_to_stop = ValueObject::eRangeOperatorExpanded;
                        *final_result = ValueObject::eValueObjectList;
                        return max_index; // tell me number of items I added to the VOList
                    }
                }
                const char *separator_position = ::strchr(expression_cstr+1,'-');
                const char *close_bracket_position = ::strchr(expression_cstr+1,']');
                if (!close_bracket_position) // if there is no ], this is a syntax error
                {
                    *first_unparsed = expression_cstr;
                    *reason_to_stop = ValueObject::eUnexpectedSymbol;
                    *final_result = ValueObject::eInvalid;
                    return 0;
                }
                if (!separator_position || separator_position > close_bracket_position) // if no separator, this is either [] or [N]
                {
                    char *end = NULL;
                    unsigned long index = ::strtoul (expression_cstr+1, &end, 0);
                    if (!end || end != close_bracket_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eUnexpectedSymbol;
                        *final_result = ValueObject::eInvalid;
                        return 0;
                    }
                    if (end - expression_cstr == 1) // if this is [], only return a valid value for arrays
                    {
                        if (root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                        {
                            int max_index = root->GetNumChildren() - 1;
                            for (int index = 0; index < max_index; index++)
                            {
                                ValueObjectSP child = 
                                root->GetChildAtIndex(index, true);
                                list->Append(child);
                            }
                            *first_unparsed = expression_cstr+2;
                            *reason_to_stop = ValueObject::eRangeOperatorExpanded;
                            *final_result = ValueObject::eValueObjectList;
                            return max_index; // tell me number of items I added to the VOList
                        }
                        else
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eEmptyRangeNotAllowed;
                            *final_result = ValueObject::eInvalid;
                            return 0;
                        }
                    }
                    // from here on we do have a valid index
                    if (root_clang_type_info.Test(ClangASTContext::eTypeIsArray))
                    {
                        root = root->GetChildAtIndex(index, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eNoSuchChild;
                            *final_result = ValueObject::eInvalid;
                            return 0;
                        }
                        else
                        {
                            list->Append(root);
                            *first_unparsed = end+1; // skip ]
                            *reason_to_stop = ValueObject::eRangeOperatorExpanded;
                            *final_result = ValueObject::eValueObjectList;
                            return 1;
                        }
                    }
                    else if (root_clang_type_info.Test(ClangASTContext::eTypeIsPointer))
                    {
                        if (*what_next == ValueObject::eDereference &&  // if this is a ptr-to-scalar, I am accessing it by index and I would have deref'ed anyway, then do it now and use this as a bitfield
                            pointee_clang_type_info.Test(ClangASTContext::eTypeIsScalar))
                        {
                            Error error;
                            root = root->Dereference(error);
                            if (error.Fail() || !root.get())
                            {
                                *first_unparsed = expression_cstr;
                                *reason_to_stop = ValueObject::eDereferencingFailed;
                                *final_result = ValueObject::eInvalid;
                                return 0;
                            }
                            else
                            {
                                *what_next = eNothing;
                                continue;
                            }
                        }
                        else
                        {
                            root = root->GetSyntheticArrayMemberFromPointer(index, true);
                            if (!root.get())
                            {
                                *first_unparsed = expression_cstr;
                                *reason_to_stop = ValueObject::eNoSuchChild;
                                *final_result = ValueObject::eInvalid;
                                return 0;
                            }
                            else
                            {
                                list->Append(root);
                                *first_unparsed = end+1; // skip ]
                                *reason_to_stop = ValueObject::eRangeOperatorExpanded;
                                *final_result = ValueObject::eValueObjectList;
                                return 1;
                            }
                        }
                    }
                    else /*if (ClangASTContext::IsScalarType(root_clang_type))*/
                    {
                        root = root->GetSyntheticBitFieldChild(index, index, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eNoSuchChild;
                            *final_result = ValueObject::eInvalid;
                            return 0;
                        }
                        else // we do not know how to expand members of bitfields, so we just return and let the caller do any further processing
                        {
                            list->Append(root);
                            *first_unparsed = end+1; // skip ]
                            *reason_to_stop = ValueObject::eRangeOperatorExpanded;
                            *final_result = ValueObject::eValueObjectList;
                            return 1;
                        }
                    }
                }
                else // we have a low and a high index
                {
                    char *end = NULL;
                    unsigned long index_lower = ::strtoul (expression_cstr+1, &end, 0);
                    if (!end || end != separator_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eUnexpectedSymbol;
                        *final_result = ValueObject::eInvalid;
                        return 0;
                    }
                    unsigned long index_higher = ::strtoul (separator_position+1, &end, 0);
                    if (!end || end != close_bracket_position) // if something weird is in our way return an error
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eUnexpectedSymbol;
                        *final_result = ValueObject::eInvalid;
                        return 0;
                    }
                    if (index_lower > index_higher) // swap indices if required
                    {
                        unsigned long temp = index_lower;
                        index_lower = index_higher;
                        index_higher = temp;
                    }
                    if (root_clang_type_info.Test(ClangASTContext::eTypeIsScalar)) // expansion only works for scalars
                    {
                        root = root->GetSyntheticBitFieldChild(index_lower, index_higher, true);
                        if (!root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eNoSuchChild;
                            *final_result = ValueObject::eInvalid;
                            return 0;
                        }
                        else
                        {
                            list->Append(root);
                            *first_unparsed = end+1; // skip ]
                            *reason_to_stop = ValueObject::eRangeOperatorExpanded;
                            *final_result = ValueObject::eValueObjectList;
                            return 1;
                        }
                    }
                    else if (root_clang_type_info.Test(ClangASTContext::eTypeIsPointer) && // if this is a ptr-to-scalar, I am accessing it by index and I would have deref'ed anyway, then do it now and use this as a bitfield
                             *what_next == ValueObject::eDereference &&
                             pointee_clang_type_info.Test(ClangASTContext::eTypeIsScalar))
                    {
                        Error error;
                        root = root->Dereference(error);
                        if (error.Fail() || !root.get())
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eDereferencingFailed;
                            *final_result = ValueObject::eInvalid;
                            return 0;
                        }
                        else
                        {
                            *what_next = ValueObject::eNothing;
                            continue;
                        }
                    }
                    else
                    {
                        for (unsigned long index = index_lower;
                             index <= index_higher; index++)
                        {
                            ValueObjectSP child = 
                                root->GetChildAtIndex(index, true);
                            list->Append(child);
                        }
                        *first_unparsed = end+1;
                        *reason_to_stop = ValueObject::eRangeOperatorExpanded;
                        *final_result = ValueObject::eValueObjectList;
                        return index_higher-index_lower+1; // tell me number of items I added to the VOList
                    }
                }
                break;
            }
            default: // some non-[ separator, or something entirely wrong, is in the way
            {
                *first_unparsed = expression_cstr;
                *reason_to_stop = ValueObject::eUnexpectedSymbol;
                *final_result = ValueObject::eInvalid;
                return 0;
                break;
            }
        }
    }
}

void
ValueObject::DumpValueObject 
(
    Stream &s,
    ValueObject *valobj,
    const char *root_valobj_name,
    uint32_t ptr_depth,
    uint32_t curr_depth,
    uint32_t max_depth,
    bool show_types,
    bool show_location,
    bool use_objc,
    lldb::DynamicValueType use_dynamic,
    bool use_synth,
    bool scope_already_checked,
    bool flat_output,
    uint32_t omit_summary_depth
)
{
    if (valobj)
    {
        bool update_success = valobj->UpdateValueIfNeeded ();

        if (update_success && use_dynamic != lldb::eNoDynamicValues)
        {
            ValueObject *dynamic_value = valobj->GetDynamicValue(use_dynamic).get();
            if (dynamic_value)
                valobj = dynamic_value;
        }
        
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
                s.Printf("%s: ", valobj->GetLocationAsCString());
            }

            s.Indent();

            // Always show the type for the top level items.
            if (show_types || (curr_depth == 0 && !flat_output))
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

            if (!scope_already_checked && !valobj->IsInScope())
            {
                err_cstr = "out of scope";
            }
        }
        
        const char *val_cstr = NULL;
        const char *sum_cstr = NULL;
        SummaryFormat* entry = valobj->GetSummaryFormat().get();
        
        if (omit_summary_depth > 0)
            entry = NULL;
        
        if (err_cstr == NULL)
        {
            val_cstr = valobj->GetValueAsCString();
            err_cstr = valobj->GetError().AsCString();
        }

        if (err_cstr)
        {
            s.Printf (" <%s>\n", err_cstr);
        }
        else
        {
            const bool is_ref = type_flags.Test (ClangASTContext::eTypeIsReference);
            if (print_valobj)
            {
                
                sum_cstr = (omit_summary_depth == 0) ? valobj->GetSummaryAsCString() : NULL;

                // We must calculate this value in realtime because entry might alter this variable's value
                // (e.g. by saying ${var%fmt}) and render precached values useless
                if (val_cstr && (!entry || entry->DoesPrintValue() || !sum_cstr))
                    s.Printf(" %s", valobj->GetValueAsCString());

                if (sum_cstr)
                {
                    // for some reason, using %@ (ObjC description) in a summary string, makes
                    // us believe we need to reset ourselves, thus invalidating the content of
                    // sum_cstr. Thus, IF we had a valid sum_cstr before, but it is now empty
                    // let us recalculate it!
                    if (sum_cstr[0] == '\0')
                        s.Printf(" %s", valobj->GetSummaryAsCString());
                    else
                        s.Printf(" %s", sum_cstr);
                }
                
                if (use_objc)
                {
                    const char *object_desc = valobj->GetObjectDescription();
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
                    AddressType ptr_address_type;
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
                
                if (print_children && (!entry || entry->DoesPrintChildren() || !sum_cstr))
                {
                    ValueObjectSP synth_vobj = valobj->GetSyntheticValue(use_synth ?
                                                                         lldb::eUseSyntheticFilter : 
                                                                         lldb::eNoSyntheticFilter);
                    const uint32_t num_children = synth_vobj->GetNumChildren();
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
                            ValueObjectSP child_sp(synth_vobj->GetChildAtIndex(idx, true));
                            if (child_sp.get())
                            {
                                DumpValueObject (s,
                                                 child_sp.get(),
                                                 NULL,
                                                 (is_ptr || is_ref) ? curr_ptr_depth - 1 : curr_ptr_depth,
                                                 curr_depth + 1,
                                                 max_depth,
                                                 show_types,
                                                 show_location,
                                                 false,
                                                 use_dynamic,
                                                 use_synth,
                                                 true,
                                                 flat_output,
                                                 omit_summary_depth > 1 ? omit_summary_depth - 1 : 0);
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
ValueObject::CreateConstantValue (const ConstString &name)
{
    ValueObjectSP valobj_sp;
    
    if (UpdateValueIfNeeded() && m_error.Success())
    {
        ExecutionContextScope *exe_scope = GetExecutionContextScope();
        if (exe_scope)
        {
            ExecutionContext exe_ctx;
            exe_scope->CalculateExecutionContext(exe_ctx);

            clang::ASTContext *ast = GetClangAST ();

            DataExtractor data;
            data.SetByteOrder (m_data.GetByteOrder());
            data.SetAddressByteSize(m_data.GetAddressByteSize());

            m_error = m_value.GetValueAsData (&exe_ctx, ast, data, 0, GetModule());

            valobj_sp = ValueObjectConstResult::Create (exe_scope, 
                                                        ast,
                                                        GetClangType(),
                                                        name,
                                                        data);
        }
    }
    
    if (!valobj_sp)
    {
        valobj_sp = ValueObjectConstResult::Create (NULL, m_error);
    }
    return valobj_sp;
}

lldb::ValueObjectSP
ValueObject::Dereference (Error &error)
{
    if (m_deref_valobj)
        return m_deref_valobj->GetSP();
        
    const bool is_pointer_type = IsPointerType();
    if (is_pointer_type)
    {
        bool omit_empty_base_classes = true;
        bool ignore_array_bounds = false;

        std::string child_name_str;
        uint32_t child_byte_size = 0;
        int32_t child_byte_offset = 0;
        uint32_t child_bitfield_bit_size = 0;
        uint32_t child_bitfield_bit_offset = 0;
        bool child_is_base_class = false;
        bool child_is_deref_of_parent = false;
        const bool transparent_pointers = false;
        clang::ASTContext *clang_ast = GetClangAST();
        clang_type_t clang_type = GetClangType();
        clang_type_t child_clang_type;

        ExecutionContext exe_ctx;
        GetExecutionContextScope()->CalculateExecutionContext (exe_ctx);
        
        child_clang_type = ClangASTContext::GetChildClangTypeAtIndex (&exe_ctx,
                                                                      clang_ast,
                                                                      GetName().GetCString(),
                                                                      clang_type,
                                                                      0,
                                                                      transparent_pointers,
                                                                      omit_empty_base_classes,
                                                                      ignore_array_bounds,
                                                                      child_name_str,
                                                                      child_byte_size,
                                                                      child_byte_offset,
                                                                      child_bitfield_bit_size,
                                                                      child_bitfield_bit_offset,
                                                                      child_is_base_class,
                                                                      child_is_deref_of_parent);
        if (child_clang_type && child_byte_size)
        {
            ConstString child_name;
            if (!child_name_str.empty())
                child_name.SetCString (child_name_str.c_str());

            m_deref_valobj = new ValueObjectChild (*this,
                                                   clang_ast,
                                                   child_clang_type,
                                                   child_name,
                                                   child_byte_size,
                                                   child_byte_offset,
                                                   child_bitfield_bit_size,
                                                   child_bitfield_bit_offset,
                                                   child_is_base_class,
                                                   child_is_deref_of_parent);
        }
    }

    if (m_deref_valobj)
    {
        error.Clear();
        return m_deref_valobj->GetSP();
    }
    else
    {
        StreamString strm;
        GetExpressionPath(strm, true);

        if (is_pointer_type)
            error.SetErrorStringWithFormat("dereference failed: (%s) %s", GetTypeName().AsCString("<invalid type>"), strm.GetString().c_str());
        else
            error.SetErrorStringWithFormat("not a pointer type: (%s) %s", GetTypeName().AsCString("<invalid type>"), strm.GetString().c_str());
        return ValueObjectSP();
    }
}

lldb::ValueObjectSP
ValueObject::AddressOf (Error &error)
{
    if (m_addr_of_valobj_sp)
        return m_addr_of_valobj_sp;
        
    AddressType address_type = eAddressTypeInvalid;
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
                    m_addr_of_valobj_sp = ValueObjectConstResult::Create (GetExecutionContextScope(),
                                                                          ast, 
                                                                          ClangASTContext::CreatePointerType (ast, clang_type),
                                                                          ConstString (name.c_str()),
                                                                          addr, 
                                                                          eAddressTypeInvalid,
                                                                          m_data.GetAddressByteSize());
                }
            }
            break;
        }
    }
    return m_addr_of_valobj_sp;
}


lldb::ValueObjectSP
ValueObject::CastPointerType (const char *name, ClangASTType &clang_ast_type)
{
    lldb::ValueObjectSP valobj_sp;
    AddressType address_type;
    const bool scalar_is_load_address = true;
    lldb::addr_t ptr_value = GetPointerValue (address_type, scalar_is_load_address);
    
    if (ptr_value != LLDB_INVALID_ADDRESS)
    {
        Address ptr_addr (NULL, ptr_value);
        
        valobj_sp = ValueObjectMemory::Create (GetExecutionContextScope(),
                                               name, 
                                               ptr_addr, 
                                               clang_ast_type);
    }
    return valobj_sp;    
}

lldb::ValueObjectSP
ValueObject::CastPointerType (const char *name, TypeSP &type_sp)
{
    lldb::ValueObjectSP valobj_sp;
    AddressType address_type;
    const bool scalar_is_load_address = true;
    lldb::addr_t ptr_value = GetPointerValue (address_type, scalar_is_load_address);
    
    if (ptr_value != LLDB_INVALID_ADDRESS)
    {
        Address ptr_addr (NULL, ptr_value);
        
        valobj_sp = ValueObjectMemory::Create (GetExecutionContextScope(),
                                               name, 
                                               ptr_addr, 
                                               type_sp);
    }
    return valobj_sp;
}


ValueObject::EvaluationPoint::EvaluationPoint () :
    m_thread_id (LLDB_INVALID_UID),
    m_stop_id (0)
{
}

ValueObject::EvaluationPoint::EvaluationPoint (ExecutionContextScope *exe_scope, bool use_selected):
    m_needs_update (true),
    m_first_update (true),
    m_thread_id (LLDB_INVALID_THREAD_ID),
    m_stop_id (0)
    
{
    ExecutionContext exe_ctx;
    ExecutionContextScope *computed_exe_scope = exe_scope;  // If use_selected is true, we may find a better scope,
                                                            // and if so we want to cache that not the original.
    if (exe_scope)                                            
        exe_scope->CalculateExecutionContext(exe_ctx);
    if (exe_ctx.target != NULL)
    {
        m_target_sp = exe_ctx.target->GetSP();
        
        if (exe_ctx.process == NULL)
            m_process_sp = exe_ctx.target->GetProcessSP();
        else
            m_process_sp = exe_ctx.process->GetSP();
        
        if (m_process_sp != NULL)
        {
            m_stop_id = m_process_sp->GetStopID();
            Thread *thread = NULL;
            
            if (exe_ctx.thread == NULL)
            {
                if (use_selected)
                {
                    thread = m_process_sp->GetThreadList().GetSelectedThread().get();
                    if (thread)
                        computed_exe_scope = thread;
                }
            }
            else 
                thread = exe_ctx.thread;
                
            if (thread != NULL)
            {
                m_thread_id = thread->GetIndexID();
                if (exe_ctx.frame == NULL)
                {
                    if (use_selected)
                    {
                        StackFrame *frame = exe_ctx.thread->GetSelectedFrame().get();
                        if (frame)
                        {
                            m_stack_id = frame->GetStackID();
                            computed_exe_scope = frame;
                        }
                    }
                }
                else
                    m_stack_id = exe_ctx.frame->GetStackID();
            }
        }
    }
    m_exe_scope = computed_exe_scope;
}

ValueObject::EvaluationPoint::EvaluationPoint (const ValueObject::EvaluationPoint &rhs) :
    m_exe_scope (rhs.m_exe_scope),
    m_needs_update(true),
    m_first_update(true),
    m_target_sp (rhs.m_target_sp),
    m_process_sp (rhs.m_process_sp),
    m_thread_id (rhs.m_thread_id),
    m_stack_id (rhs.m_stack_id),
    m_stop_id (0)
{
}

ValueObject::EvaluationPoint::~EvaluationPoint () 
{
}

ExecutionContextScope *
ValueObject::EvaluationPoint::GetExecutionContextScope ()
{
    // We have to update before giving out the scope, or we could be handing out stale pointers.    
    SyncWithProcessState();
    
    return m_exe_scope;
}

// This function checks the EvaluationPoint against the current process state.  If the current
// state matches the evaluation point, or the evaluation point is already invalid, then we return
// false, meaning "no change".  If the current state is different, we update our state, and return
// true meaning "yes, change".  If we did see a change, we also set m_needs_update to true, so 
// future calls to NeedsUpdate will return true.

bool
ValueObject::EvaluationPoint::SyncWithProcessState()
{
    // If we're already invalid, we don't need to do anything, and nothing has changed:
    if (m_stop_id == LLDB_INVALID_UID)
    {
        // Can't update with an invalid state.
        m_needs_update = false;
        return false;
    }
    
    // If we don't have a process nothing can change.
    if (!m_process_sp)
        return false;
        
    // If our stop id is the current stop ID, nothing has changed:
    uint32_t cur_stop_id = m_process_sp->GetStopID();
    if (m_stop_id == cur_stop_id)
        return false;
    
    // If the current stop id is 0, either we haven't run yet, or the process state has been cleared.
    // In either case, we aren't going to be able to sync with the process state.
    if (cur_stop_id == 0)
        return false;
        
    m_stop_id = cur_stop_id;
    m_needs_update = true;
    m_exe_scope = m_process_sp.get();
    
    // Something has changed, so we will return true.  Now make sure the thread & frame still exist, and if either
    // doesn't, mark ourselves as invalid.
    
    if (m_thread_id != LLDB_INVALID_THREAD_ID)
    {
        Thread *our_thread = m_process_sp->GetThreadList().FindThreadByIndexID (m_thread_id).get();
        if (our_thread == NULL)
        {
            SetInvalid();
        }
        else
        {
            m_exe_scope = our_thread;
            
            if (m_stack_id.IsValid())
            {
                StackFrame *our_frame = our_thread->GetFrameWithStackID (m_stack_id).get();
                if (our_frame == NULL)
                    SetInvalid();
                else
                    m_exe_scope = our_frame;
            }
        }
    }
    return true;
}

void
ValueObject::EvaluationPoint::SetUpdated ()
{
    m_first_update = false;
    m_needs_update = false;
    if (m_process_sp)
        m_stop_id = m_process_sp->GetStopID();
}
        

bool
ValueObject::EvaluationPoint::SetContext (ExecutionContextScope *exe_scope)
{
    if (!IsValid())
        return false;
    
    bool needs_update = false;
    m_exe_scope = NULL;
    
    // The target has to be non-null, and the 
    Target *target = exe_scope->CalculateTarget();
    if (target != NULL)
    {
        Target *old_target = m_target_sp.get();
        assert (target == old_target);
        Process *process = exe_scope->CalculateProcess();
        if (process != NULL)
        {
            // FOR NOW - assume you can't update variable objects across process boundaries.
            Process *old_process = m_process_sp.get();
            assert (process == old_process);
            
            lldb::user_id_t stop_id = process->GetStopID();
            if (stop_id != m_stop_id)
            {
                needs_update = true;
                m_stop_id = stop_id;
            }
            // See if we're switching the thread or stack context.  If no thread is given, this is
            // being evaluated in a global context.            
            Thread *thread = exe_scope->CalculateThread();
            if (thread != NULL)
            {
                lldb::user_id_t new_thread_index = thread->GetIndexID();
                if (new_thread_index != m_thread_id)
                {
                    needs_update = true;
                    m_thread_id = new_thread_index;
                    m_stack_id.Clear();
                }
                
                StackFrame *new_frame = exe_scope->CalculateStackFrame();
                if (new_frame != NULL)
                {
                    if (new_frame->GetStackID() != m_stack_id)
                    {
                        needs_update = true;
                        m_stack_id = new_frame->GetStackID();
                    }
                }
                else
                {
                    m_stack_id.Clear();
                    needs_update = true;
                }
            }
            else
            {
                // If this had been given a thread, and now there is none, we should update.
                // Otherwise we don't have to do anything.
                if (m_thread_id != LLDB_INVALID_UID)
                {
                    m_thread_id = LLDB_INVALID_UID;
                    m_stack_id.Clear();
                    needs_update = true;
                }
            }
        }
        else
        {
            // If there is no process, then we don't need to update anything.
            // But if we're switching from having a process to not, we should try to update.
            if (m_process_sp.get() != NULL)
            {
                needs_update = true;
                m_process_sp.reset();
                m_thread_id = LLDB_INVALID_UID;
                m_stack_id.Clear();
            }
        }
    }
    else
    {
        // If there's no target, nothing can change so we don't need to update anything.
        // But if we're switching from having a target to not, we should try to update.
        if (m_target_sp.get() != NULL)
        {
            needs_update = true;
            m_target_sp.reset();
            m_process_sp.reset();
            m_thread_id = LLDB_INVALID_UID;
            m_stack_id.Clear();
        }
    }
    if (!m_needs_update)
        m_needs_update = needs_update;
        
    return needs_update;
}

void
ValueObject::ClearUserVisibleData()
{
    m_location_str.clear();
    m_value_str.clear();
    m_summary_str.clear();
    m_object_desc_str.clear();
}
