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
#include "lldb/Core/DataVisualization.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectChild.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectDynamicValue.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Core/ValueObjectSyntheticFilter.h"

#include "lldb/Host/Endian.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "lldb/Utility/RefCounter.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_utility;

static user_id_t g_value_obj_uid = 0;

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
    m_last_format_mgr_dynamic(parent.m_last_format_mgr_dynamic),
    m_last_summary_format(),
    m_forced_summary_format(),
    m_last_value_format(),
    m_last_synthetic_filter(),
    m_user_id_of_forced_summary(),
    m_value_is_valid (false),
    m_value_did_change (false),
    m_children_count_valid (false),
    m_old_value_valid (false),
    m_pointers_point_to_load_addrs (false),
    m_is_deref_of_parent (false),
    m_is_array_item_for_pointer(false),
    m_is_bitfield_for_scalar(false),
    m_is_expression_path_child(false),
    m_is_child_at_offset(false),
    m_is_expression_result(parent.m_is_expression_result),
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
    m_last_format_mgr_dynamic(eNoDynamicValues),
    m_last_summary_format(),
    m_forced_summary_format(),
    m_last_value_format(),
    m_last_synthetic_filter(),
    m_user_id_of_forced_summary(),
    m_value_is_valid (false),
    m_value_did_change (false),
    m_children_count_valid (false),
    m_old_value_valid (false),
    m_pointers_point_to_load_addrs (false),
    m_is_deref_of_parent (false),
    m_is_array_item_for_pointer(false),
    m_is_bitfield_for_scalar(false),
    m_is_expression_path_child(false),
    m_is_child_at_offset(false),
    m_is_expression_result(false),
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
    return UpdateValueIfNeeded(m_last_format_mgr_dynamic, update_format);
}

bool
ValueObject::UpdateValueIfNeeded (DynamicValueType use_dynamic, bool update_format)
{
    
    if (update_format)
        UpdateFormatsIfNeeded(use_dynamic);
    
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
ValueObject::UpdateFormatsIfNeeded(DynamicValueType use_dynamic)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    if (log)
        log->Printf("checking for FormatManager revisions. VO named %s is at revision %d, while the format manager is at revision %d",
           GetName().GetCString(),
           m_last_format_mgr_revision,
           DataVisualization::GetCurrentRevision());
    if (HasCustomSummaryFormat() && m_update_point.GetModID() != m_user_id_of_forced_summary)
    {
        ClearCustomSummaryFormat();
        m_summary_str.clear();
    }
    if ( (m_last_format_mgr_revision != DataVisualization::GetCurrentRevision()) ||
          m_last_format_mgr_dynamic != use_dynamic)
    {
        if (m_last_summary_format.get())
            m_last_summary_format.reset((StringSummaryFormat*)NULL);
        if (m_last_value_format.get())
            m_last_value_format.reset(/*(ValueFormat*)NULL*/);
        if (m_last_synthetic_filter.get())
            m_last_synthetic_filter.reset(/*(SyntheticFilter*)NULL*/);

        m_synthetic_value = NULL;
        
        DataVisualization::ValueFormats::Get(*this, eNoDynamicValues, m_last_value_format);
        DataVisualization::GetSummaryFormat(*this, use_dynamic, m_last_summary_format);
        DataVisualization::GetSyntheticChildren(*this, use_dynamic, m_last_synthetic_filter);

        m_last_format_mgr_revision = DataVisualization::GetCurrentRevision();
        m_last_format_mgr_dynamic = use_dynamic;

        ClearUserVisibleData();
    }
}

void
ValueObject::SetNeedsUpdate ()
{
    m_update_point.SetNeedsUpdate();
    // We have to clear the value string here so ConstResult children will notice if their values are
    // changed by hand (i.e. with SetValueAsCString).
    m_value_str.clear();
}

DataExtractor &
ValueObject::GetDataExtractor ()
{
    UpdateValueIfNeeded(false);
    return m_data;
}

const Error &
ValueObject::GetError()
{
    UpdateValueIfNeeded(false);
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
    if (UpdateValueIfNeeded(false))
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
    if (UpdateValueIfNeeded(false)) // make sure that you are up to date before returning anything
    {
        ExecutionContext exe_ctx;
        ExecutionContextScope *exe_scope = GetExecutionContextScope();
        if (exe_scope)
            exe_scope->CalculateExecutionContext(exe_ctx);
        Value tmp_value(m_value);
        scalar = tmp_value.ResolveValue(&exe_ctx, GetClangAST ());
        return scalar.IsValid();
    }
    else
        return false;
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
        UpdateValueIfNeeded(false);
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
        UpdateValueIfNeeded(false);

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
    if (UpdateValueIfNeeded (true))
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
                            addr_t func_ptr_address = GetPointerValue (func_ptr_address_type, true);

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
    addr_t cstr_address = LLDB_INVALID_ADDRESS;
    AddressType cstr_address_type = eAddressTypeInvalid;
    cstr_address = GetAddressOf (cstr_address_type, true);
    return (cstr_address != LLDB_INVALID_ADDRESS);
}

void
ValueObject::ReadPointedString(Stream& s,
                               Error& error,
                               uint32_t max_length,
                               bool honor_array,
                               Format item_format)
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
                if (target == NULL)
                {
                    s << "<no target to read from>";
                }
                else
                {
                    addr_t cstr_address = LLDB_INVALID_ADDRESS;
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
                    if (cstr_address == LLDB_INVALID_ADDRESS)
                    {
                        s << "<invalid address for data>";
                    }
                    else
                    {
                        Address cstr_so_addr (NULL, cstr_address);
                        DataExtractor data;
                        size_t bytes_read = 0;
                        std::vector<char> data_buffer;
                        bool prefer_file_cache = false;
                        if (cstr_len > 0 && honor_array)
                        {
                            data_buffer.resize(cstr_len);
                            data.SetData (&data_buffer.front(), data_buffer.size(), endian::InlHostByteOrder());
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
                            else
                                s << "\"<data not available>\"";
                        }
                        else
                        {
                            cstr_len = max_length;
                            const size_t k_max_buf_size = 64;
                            data_buffer.resize (k_max_buf_size + 1);
                            // NULL terminate in case we don't get the entire C string
                            data_buffer.back() = '\0';
                            
                            s << '"';
                            
                            bool any_data = false;
                            
                            data.SetData (&data_buffer.front(), data_buffer.size(), endian::InlHostByteOrder());
                            while ((bytes_read = target->ReadMemory (cstr_so_addr, 
                                                                     prefer_file_cache,
                                                                     &data_buffer.front(), 
                                                                     k_max_buf_size, 
                                                                     error)) > 0)
                            {
                                any_data = true;
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
                            
                            if (any_data == false)
                                s << "<data not available>";
                            
                            s << '"';
                        }
                    }
                }
            }
    }
    else
    {
        error.SetErrorString("impossible to read a string from this object");
        s << "<not a string object>";
    }
}

const char *
ValueObject::GetObjectDescription ()
{
    
    if (!UpdateValueIfNeeded (true))
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
    
    LanguageType language = GetObjectRuntimeLanguage();
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
                runtime = process->GetLanguageRuntime(eLanguageTypeObjC);
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
        if (UpdateValueIfNeeded(true))
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
                            if (m_format == lldb::eFormatDefault && m_last_value_format)
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

// if > 8bytes, 0 is returned. this method should mostly be used
// to read address values out of pointers
uint64_t
ValueObject::GetValueAsUnsigned (uint64_t fail_value)
{
    // If our byte size is zero this is an aggregate type that has children
    if (ClangASTContext::IsAggregateType (GetClangType()) == false)
    {
        Scalar scalar;
        if (ResolveValue (scalar))
            return scalar.GetRawBits64(fail_value);
    }
    return fail_value;
}

bool
ValueObject::GetPrintableRepresentation(Stream& s,
                                        ValueObjectRepresentationStyle val_obj_display,
                                        Format custom_format)
{

    RefCounter ref(&m_dump_printable_counter);
    
    if (custom_format != eFormatInvalid)
        SetFormat(custom_format);
    
    const char * return_value;
    std::string alloc_mem;
    
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
        case eDisplayChildrenCount:
        {
            alloc_mem.resize(512);
            return_value = &alloc_mem[0];
            int count = GetNumChildren();
            snprintf((char*)return_value, 512, "%d", count);
            break;
        }
        case eDisplayType:
            return_value = GetTypeName().AsCString();
            break;
        default:
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
                // this thing has no value, and it seems to have no summary
                // some combination of unitialized data and other factors can also
                // raise this condition, so let's print a nice generic error message
                {
                    alloc_mem.resize(684);
                    return_value = &alloc_mem[0];
                    snprintf((char*)return_value, 684, "%s @ %s", GetTypeName().AsCString(), GetLocationAsCString());
                }
            }
            else
                return_value = GetValueAsCString();
        }
    }
    
    if (return_value)
        s.PutCString(return_value);
    else
    {
        if (m_error.Fail())
            s.Printf("<%s>", m_error.AsCString());
        else if (val_obj_display == eDisplaySummary)
            s.PutCString("<no summary available>");
        else if (val_obj_display == eDisplayValue)
            s.PutCString("<no value available>");
        else if (val_obj_display == eDisplayLanguageSpecific)
            s.PutCString("<not a valid Objective-C object>"); // edit this if we have other runtimes that support a description
        else
            s.PutCString("<no printable representation>");
    }
    
    // we should only return false here if we could not do *anything*
    // even if we have an error message as output, that's a success
    // from our callers' perspective, so return true
    return true;
    
}

// if any more "special cases" are added to ValueObject::DumpPrintableRepresentation() please keep
// this call up to date by returning true for your new special cases. We will eventually move
// to checking this call result before trying to display special cases
bool
ValueObject::HasSpecialCasesForPrintableRepresentation(ValueObjectRepresentationStyle val_obj_display,
                                                       Format custom_format)
{
    clang_type_t elem_or_pointee_type;
    Flags flags(ClangASTContext::GetTypeInfo(GetClangType(), GetClangAST(), &elem_or_pointee_type));
    
    if (flags.AnySet(ClangASTContext::eTypeIsArray | ClangASTContext::eTypeIsPointer)
        && val_obj_display == ValueObject::eDisplayValue)
    {        
        if (IsCStringContainer(true) && 
            (custom_format == eFormatCString ||
             custom_format == eFormatCharArray ||
             custom_format == eFormatChar ||
             custom_format == eFormatVectorOfChar))
            return true;

        if (flags.Test(ClangASTContext::eTypeIsArray))
        {
            if ((custom_format == eFormatBytes) ||
                (custom_format == eFormatBytesWithASCII))
                return true;
            
            if ((custom_format == eFormatVectorOfChar) ||
                (custom_format == eFormatVectorOfFloat32) ||
                (custom_format == eFormatVectorOfFloat64) ||
                (custom_format == eFormatVectorOfSInt16) ||
                (custom_format == eFormatVectorOfSInt32) ||
                (custom_format == eFormatVectorOfSInt64) ||
                (custom_format == eFormatVectorOfSInt8) ||
                (custom_format == eFormatVectorOfUInt128) ||
                (custom_format == eFormatVectorOfUInt16) ||
                (custom_format == eFormatVectorOfUInt32) ||
                (custom_format == eFormatVectorOfUInt64) ||
                (custom_format == eFormatVectorOfUInt8))
                return true;
        }
    }
    return false;
}

bool
ValueObject::DumpPrintableRepresentation(Stream& s,
                                         ValueObjectRepresentationStyle val_obj_display,
                                         Format custom_format,
                                         bool only_special)
{

    clang_type_t elem_or_pointee_type;
    Flags flags(ClangASTContext::GetTypeInfo(GetClangType(), GetClangAST(), &elem_or_pointee_type));
    
    if (flags.AnySet(ClangASTContext::eTypeIsArray | ClangASTContext::eTypeIsPointer)
         && val_obj_display == ValueObject::eDisplayValue)
    {
        // when being asked to get a printable display an array or pointer type directly, 
        // try to "do the right thing"
        
        if (IsCStringContainer(true) && 
            (custom_format == eFormatCString ||
             custom_format == eFormatCharArray ||
             custom_format == eFormatChar ||
             custom_format == eFormatVectorOfChar)) // print char[] & char* directly
        {
            Error error;
            ReadPointedString(s,
                              error,
                              0,
                              (custom_format == eFormatVectorOfChar) ||
                              (custom_format == eFormatCharArray));
            return !error.Fail();
        }
        
        if (custom_format == eFormatEnum)
            return false;
        
        // this only works for arrays, because I have no way to know when
        // the pointed memory ends, and no special \0 end of data marker
        if (flags.Test(ClangASTContext::eTypeIsArray))
        {
            if ((custom_format == eFormatBytes) ||
                (custom_format == eFormatBytesWithASCII))
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
            
            if ((custom_format == eFormatVectorOfChar) ||
                (custom_format == eFormatVectorOfFloat32) ||
                (custom_format == eFormatVectorOfFloat64) ||
                (custom_format == eFormatVectorOfSInt16) ||
                (custom_format == eFormatVectorOfSInt32) ||
                (custom_format == eFormatVectorOfSInt64) ||
                (custom_format == eFormatVectorOfSInt8) ||
                (custom_format == eFormatVectorOfUInt128) ||
                (custom_format == eFormatVectorOfUInt16) ||
                (custom_format == eFormatVectorOfUInt32) ||
                (custom_format == eFormatVectorOfUInt64) ||
                (custom_format == eFormatVectorOfUInt8)) // arrays of bytes, bytes with ASCII or any vector format should be printed directly
            {
                uint32_t count = GetNumChildren();

                Format format = FormatManager::GetSingleItemFormat(custom_format);
                
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
        
        if ((custom_format == eFormatBoolean) ||
            (custom_format == eFormatBinary) ||
            (custom_format == eFormatChar) ||
            (custom_format == eFormatCharPrintable) ||
            (custom_format == eFormatComplexFloat) ||
            (custom_format == eFormatDecimal) ||
            (custom_format == eFormatHex) ||
            (custom_format == eFormatFloat) ||
            (custom_format == eFormatOctal) ||
            (custom_format == eFormatOSType) ||
            (custom_format == eFormatUnicode16) ||
            (custom_format == eFormatUnicode32) ||
            (custom_format == eFormatUnsigned) ||
            (custom_format == eFormatPointer) ||
            (custom_format == eFormatComplexInteger) ||
            (custom_format == eFormatComplex) ||
            (custom_format == eFormatDefault)) // use the [] operator
            return false;
    }
    
    if (only_special)
        return false;
    
    bool var_success = GetPrintableRepresentation(s, val_obj_display, custom_format);
    if (custom_format != eFormatInvalid)
        SetFormat(eFormatDefault);
    return var_success;
}

addr_t
ValueObject::GetAddressOf (AddressType &address_type, bool scalar_is_load_address)
{
    if (!UpdateValueIfNeeded(false))
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
    addr_t address = LLDB_INVALID_ADDRESS;
    address_type = eAddressTypeInvalid;
    
    if (!UpdateValueIfNeeded(false))
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
    if (!UpdateValueIfNeeded(false))
        return false;

    uint32_t count = 0;
    Encoding encoding = ClangASTType::GetEncoding (GetClangType(), count);

    const size_t byte_size = GetByteSize();

    Value::ValueType value_type = m_value.GetValueType();
    
    if (value_type == Value::eValueTypeScalar)
    {
        // If the value is already a scalar, then let the scalar change itself:
        m_value.GetScalar().SetValueFromCString (value_str, encoding, byte_size);
    }
    else if (byte_size <= Scalar::GetMaxByteSize())
    {
        // If the value fits in a scalar, then make a new scalar and again let the
        // scalar code do the conversion, then figure out where to put the new value.
        Scalar new_scalar;
        Error error;
        error = new_scalar.SetValueFromCString (value_str, encoding, byte_size);
        if (error.Success())
        {
            switch (value_type)
            {
                case Value::eValueTypeLoadAddress:
                {
                    // If it is a load address, then the scalar value is the storage location
                    // of the data, and we have to shove this value down to that load location.
                    ProcessSP process_sp = GetUpdatePoint().GetProcessSP();
                    if (process_sp)
                    {
                        addr_t target_addr = m_value.GetScalar().GetRawBits64(LLDB_INVALID_ADDRESS);
                        size_t bytes_written = process_sp->WriteScalarToMemory (target_addr, 
                                                                            new_scalar, 
                                                                            byte_size, 
                                                                            error);
                        if (!error.Success() || bytes_written != byte_size)
                            return false;                            
                    }
                }
                break;
                case Value::eValueTypeHostAddress:
                {
                    // If it is a host address, then we stuff the scalar as a DataBuffer into the Value's data.
                    DataExtractor new_data;
                    new_data.SetByteOrder (m_data.GetByteOrder());
                    
                    DataBufferSP buffer_sp (new DataBufferHeap(byte_size, 0));
                    m_data.SetData(buffer_sp, 0);
                    bool success = new_scalar.GetData(new_data);
                    if (success)
                    {
                        new_data.CopyByteOrderedData(0, 
                                                     byte_size, 
                                                     const_cast<uint8_t *>(m_data.GetDataStart()), 
                                                     byte_size, 
                                                     m_data.GetByteOrder());
                    }
                    m_value.GetScalar() = (uintptr_t)m_data.GetDataStart();
                    
                }
                break;
                case Value::eValueTypeFileAddress:
                case Value::eValueTypeScalar:
                break;    
            }
        }
        else
        {
            return false;
        }
    }
    else
    {
        // We don't support setting things bigger than a scalar at present.
        return false;
    }
    
    // If we have reached this point, then we have successfully changed the value.
    SetNeedsUpdate();
    return true;
}

LanguageType
ValueObject::GetObjectRuntimeLanguage ()
{
    return ClangASTType::GetMinimumLanguage (GetClangAST(),
                                             GetClangType());
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
ValueObject::GetSyntheticArrayMember (int32_t index, bool can_create)
{
    if (IsArrayType())
        return GetSyntheticArrayMemberFromArray(index, can_create);

    if (IsPointerType())
        return GetSyntheticArrayMemberFromPointer(index, can_create);
    
    return ValueObjectSP();
    
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
                synthetic_child_sp->SetName(ConstString(index_str));
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
                synthetic_child_sp->SetName(ConstString(index_str));
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
                synthetic_child_sp->SetName(ConstString(index_str));
                synthetic_child_sp->m_is_bitfield_for_scalar = true;
            }
        }
    }
    return synthetic_child_sp;
}

ValueObjectSP
ValueObject::GetSyntheticArrayRangeChild (uint32_t from, uint32_t to, bool can_create)
{
    ValueObjectSP synthetic_child_sp;
    if (IsArrayType () || IsPointerType ())
    {
        char index_str[64];
        snprintf(index_str, sizeof(index_str), "[%i-%i]", from, to);
        ConstString index_const_str(index_str);
        // Check if we have already created a synthetic array member in this
        // valid object. If we have we will re-use it.
        synthetic_child_sp = GetSyntheticChild (index_const_str);
        if (!synthetic_child_sp)
        {
            ValueObjectSynthetic *synthetic_child;
            
            // We haven't made a synthetic array member for INDEX yet, so
            // lets make one and cache it for any future reference.
            SyntheticArrayView *view = new SyntheticArrayView();
            view->AddRange(from,to);
            SyntheticChildrenSP view_sp(view);
            synthetic_child = new ValueObjectSynthetic(*this, view_sp);
            
            // Cache the value if we got one back...
            if (synthetic_child)
            {
                AddSyntheticChild(index_const_str, synthetic_child);
                synthetic_child_sp = synthetic_child->GetSP();
                synthetic_child_sp->SetName(ConstString(index_str));
                synthetic_child_sp->m_is_bitfield_for_scalar = true;
            }
        }
    }
    return synthetic_child_sp;
}

ValueObjectSP
ValueObject::GetSyntheticChildAtOffset(uint32_t offset, const ClangASTType& type, bool can_create)
{
    
    ValueObjectSP synthetic_child_sp;
    
    char name_str[64];
    snprintf(name_str, sizeof(name_str), "@%i", offset);
    ConstString name_const_str(name_str);
    
    // Check if we have already created a synthetic array member in this
    // valid object. If we have we will re-use it.
    synthetic_child_sp = GetSyntheticChild (name_const_str);
    
    if (synthetic_child_sp.get())
        return synthetic_child_sp;
    
    if (!can_create)
        return ValueObjectSP();
    
    ValueObjectChild *synthetic_child = new ValueObjectChild(*this,
                                                             type.GetASTContext(),
                                                             type.GetOpaqueQualType(),
                                                             name_const_str,
                                                             type.GetTypeByteSize(),
                                                             offset,
                                                             0,
                                                             0,
                                                             false,
                                                             false);
    if (synthetic_child)
    {
        AddSyntheticChild(name_const_str, synthetic_child);
        synthetic_child_sp = synthetic_child->GetSP();
        synthetic_child_sp->SetName(name_const_str);
        synthetic_child_sp->m_is_child_at_offset = true;
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

ValueObjectSP
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
            synthetic_child_sp->SetName(ConstString(SkipLeadingExpressionPathSeparators(expression)));
            synthetic_child_sp->m_is_expression_path_child = true;
        }
    }
    return synthetic_child_sp;
}

void
ValueObject::CalculateSyntheticValue (SyntheticValueType use_synthetic)
{
    if (use_synthetic == eNoSyntheticFilter)
        return;
    
    UpdateFormatsIfNeeded(m_last_format_mgr_dynamic);
    
    if (m_last_synthetic_filter.get() == NULL)
        return;
    
    if (m_synthetic_value == NULL)
        m_synthetic_value = new ValueObjectSynthetic(*this, m_last_synthetic_filter);
    
}

void
ValueObject::CalculateDynamicValue (DynamicValueType use_dynamic)
{
    if (use_dynamic == eNoDynamicValues)
        return;
        
    if (!m_dynamic_value && !IsDynamic())
    {
        Process *process = m_update_point.GetProcessSP().get();
        bool worth_having_dynamic_value = false;
        
        
        // FIXME: Process should have some kind of "map over Runtimes" so we don't have to
        // hard code this everywhere.
        LanguageType known_type = GetObjectRuntimeLanguage();
        if (known_type != eLanguageTypeUnknown && known_type != eLanguageTypeC)
        {
            LanguageRuntime *runtime = process->GetLanguageRuntime (known_type);
            if (runtime)
                worth_having_dynamic_value = runtime->CouldHaveDynamicValue(*this);
        }
        else
        {
            LanguageRuntime *cpp_runtime = process->GetLanguageRuntime (eLanguageTypeC_plus_plus);
            if (cpp_runtime)
                worth_having_dynamic_value = cpp_runtime->CouldHaveDynamicValue(*this);
            
            if (!worth_having_dynamic_value)
            {
                LanguageRuntime *objc_runtime = process->GetLanguageRuntime (eLanguageTypeObjC);
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
    if (use_dynamic == eNoDynamicValues)
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
    if (use_synthetic == eNoSyntheticFilter)
        return GetSP();
    
    UpdateFormatsIfNeeded(m_last_format_mgr_dynamic);
    
    if (m_last_synthetic_filter.get() == NULL)
        return GetSP();
    
    CalculateSyntheticValue(use_synthetic);
    
    if (m_synthetic_value)
        return m_synthetic_value->GetSP();
    else
        return GetSP();
}

bool
ValueObject::HasSyntheticValue()
{
    UpdateFormatsIfNeeded(m_last_format_mgr_dynamic);
    
    if (m_last_synthetic_filter.get() == NULL)
        return false;
    
    CalculateSyntheticValue(eUseSyntheticFilter);
    
    if (m_synthetic_value)
        return true;
    else
        return false;
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

    if (is_deref_of_parent && epformat == eDereferencePointers)
    {
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
    
    if (is_deref_of_parent && epformat == eDereferencePointers)
    {
        s.PutChar(')');
    }
}

ValueObjectSP
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
                                        ValueObjectListSP& list,
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

ValueObjectSP
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
        
        clang_type_t root_clang_type = root->GetClangType();
        clang_type_t pointee_clang_type;
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
                    ValueObjectSP child_valobj_sp = root->GetChildMemberWithName(child_name, true);
                    
                    if (child_valobj_sp.get()) // we know we are done, so just return
                    {
                        *first_unparsed = '\0';
                        *reason_to_stop = ValueObject::eEndOfString;
                        *final_result = ValueObject::ePlain;
                        return child_valobj_sp;
                    }
                    else if (options.m_no_synthetic_children == false) // let's try with synthetic children
                    {
                        child_valobj_sp = root->GetSyntheticValue(eUseSyntheticFilter)->GetChildMemberWithName(child_name, true);
                    }
                    
                    // if we are here and options.m_no_synthetic_children is true, child_valobj_sp is going to be a NULL SP,
                    // so we hit the "else" branch, and return an error
                    if(child_valobj_sp.get()) // if it worked, just return
                    {
                        *first_unparsed = '\0';
                        *reason_to_stop = ValueObject::eEndOfString;
                        *final_result = ValueObject::ePlain;
                        return child_valobj_sp;
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
                    ValueObjectSP child_valobj_sp = root->GetChildMemberWithName(child_name, true);
                    if (child_valobj_sp.get()) // store the new root and move on
                    {
                        root = child_valobj_sp;
                        *first_unparsed = next_separator;
                        *final_result = ValueObject::ePlain;
                        continue;
                    }
                    else if (options.m_no_synthetic_children == false) // let's try with synthetic children
                    {
                        child_valobj_sp = root->GetSyntheticValue(eUseSyntheticFilter)->GetChildMemberWithName(child_name, true);
                    }
                    
                    // if we are here and options.m_no_synthetic_children is true, child_valobj_sp is going to be a NULL SP,
                    // so we hit the "else" branch, and return an error
                    if(child_valobj_sp.get()) // if it worked, move on
                    {
                        root = child_valobj_sp;
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
                    if (!root_clang_type_info.Test(ClangASTContext::eTypeIsScalar)) // if this is not even a scalar...
                    {
                        if (options.m_no_synthetic_children) // ...only chance left is synthetic
                        {
                            *first_unparsed = expression_cstr;
                            *reason_to_stop = ValueObject::eRangeOperatorInvalid;
                            *final_result = ValueObject::eInvalid;
                            return ValueObjectSP();
                        }
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
                        if (!child_valobj_sp)
                            if (root->HasSyntheticValue() && root->GetSyntheticValue(eUseSyntheticFilter)->GetNumChildren() > index)
                                child_valobj_sp = root->GetSyntheticValue(eUseSyntheticFilter)->GetChildAtIndex(index, true);
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
                            if (ClangASTType::GetMinimumLanguage(root->GetClangAST(),
                                                                    root->GetClangType()) == eLanguageTypeObjC
                                &&
                                ClangASTContext::IsPointerType(ClangASTType::GetPointeeType(root->GetClangType())) == false
                                &&
                                root->HasSyntheticValue()
                                &&
                                options.m_no_synthetic_children == false)
                            {
                                root = root->GetSyntheticValue(eUseSyntheticFilter)->GetChildAtIndex(index, true);
                            }
                            else
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
                    else if (ClangASTContext::IsScalarType(root_clang_type))
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
                    else if (root->HasSyntheticValue() && options.m_no_synthetic_children == false)
                    {
                        root = root->GetSyntheticValue(eUseSyntheticFilter)->GetChildAtIndex(index, true);
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
                    else
                    {
                        *first_unparsed = expression_cstr;
                        *reason_to_stop = ValueObject::eNoSuchChild;
                        *final_result = ValueObject::eInvalid;
                        return ValueObjectSP();
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
                                        ValueObjectSP root,
                                        ValueObjectListSP& list,
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
        
        clang_type_t root_clang_type = root->GetClangType();
        clang_type_t pointee_clang_type;
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
    DynamicValueType use_dynamic,
    bool use_synth,
    bool scope_already_checked,
    bool flat_output,
    uint32_t omit_summary_depth,
    bool ignore_cap
)
{
    if (valobj)
    {
        bool update_success = valobj->UpdateValueIfNeeded (use_dynamic, true);

        if (update_success && use_dynamic != eNoDynamicValues)
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
            {
                const char* typeName = valobj->GetTypeName().AsCString("<invalid type>");
                s.Printf("(%s", typeName);
                // only show dynamic types if the user really wants to see types
                if (show_types && use_dynamic != eNoDynamicValues &&
                    (/*strstr(typeName, "id") == typeName ||*/
                     ClangASTType::GetMinimumLanguage(valobj->GetClangAST(), valobj->GetClangType()) == eLanguageTypeObjC))
                {
                    Process* process = valobj->GetUpdatePoint().GetProcessSP().get();
                    if (process == NULL)
                        s.Printf(", dynamic type: unknown) ");
                    else
                    {
                        ObjCLanguageRuntime *runtime = process->GetObjCLanguageRuntime();
                        if (runtime == NULL)
                            s.Printf(", dynamic type: unknown) ");
                        else
                        {
                            ObjCLanguageRuntime::ObjCISA isa = runtime->GetISA(*valobj);
                            if (!runtime->IsValidISA(isa))
                                s.Printf(", dynamic type: unknown) ");
                            else
                                s.Printf(", dynamic type: %s) ",
                                         runtime->GetActualTypeName(isa).GetCString());
                        }
                    }
                }
                else
                    s.Printf(") ");
            }


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
                    ValueObjectSP synth_valobj = valobj->GetSyntheticValue(use_synth ?
                                                                         eUseSyntheticFilter : 
                                                                         eNoSyntheticFilter);
                    uint32_t num_children = synth_valobj->GetNumChildren();
                    bool print_dotdotdot = false;
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
                        
                        uint32_t max_num_children = valobj->GetUpdatePoint().GetTargetSP()->GetMaximumNumberOfChildrenToDisplay();
                        
                        if (num_children > max_num_children && !ignore_cap)
                        {
                            num_children = max_num_children;
                            print_dotdotdot = true;
                        }

                        for (uint32_t idx=0; idx<num_children; ++idx)
                        {
                            ValueObjectSP child_sp(synth_valobj->GetChildAtIndex(idx, true));
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
                                                 omit_summary_depth > 1 ? omit_summary_depth - 1 : 0,
                                                 ignore_cap);
                            }
                        }

                        if (!flat_output)
                        {
                            if (print_dotdotdot)
                            {
                                valobj->GetUpdatePoint().GetTargetSP()->GetDebugger().GetCommandInterpreter().ChildrenTruncated();
                                s.Indent("...\n");
                            }
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
    
    if (UpdateValueIfNeeded(false) && m_error.Success())
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

ValueObjectSP
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

ValueObjectSP
ValueObject::AddressOf (Error &error)
{
    if (m_addr_of_valobj_sp)
        return m_addr_of_valobj_sp;
        
    AddressType address_type = eAddressTypeInvalid;
    const bool scalar_is_load_address = false;
    addr_t addr = GetAddressOf (address_type, scalar_is_load_address);
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


ValueObjectSP
ValueObject::CastPointerType (const char *name, ClangASTType &clang_ast_type)
{
    ValueObjectSP valobj_sp;
    AddressType address_type;
    const bool scalar_is_load_address = true;
    addr_t ptr_value = GetPointerValue (address_type, scalar_is_load_address);
    
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

ValueObjectSP
ValueObject::CastPointerType (const char *name, TypeSP &type_sp)
{
    ValueObjectSP valobj_sp;
    AddressType address_type;
    const bool scalar_is_load_address = true;
    addr_t ptr_value = GetPointerValue (address_type, scalar_is_load_address);
    
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
    m_mod_id ()
{
}

ValueObject::EvaluationPoint::EvaluationPoint (ExecutionContextScope *exe_scope, bool use_selected):
    m_needs_update (true),
    m_first_update (true),
    m_thread_id (LLDB_INVALID_THREAD_ID),
    m_mod_id ()
    
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
            m_mod_id = m_process_sp->GetModID();
            
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
    m_mod_id ()
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
    // If we don't have a process nothing can change.
    if (!m_process_sp)
    {
        m_exe_scope = m_target_sp.get();
        return false;
    }
        
    // If our stop id is the current stop ID, nothing has changed:
    ProcessModID current_mod_id = m_process_sp->GetModID();
    
    // If the current stop id is 0, either we haven't run yet, or the process state has been cleared.
    // In either case, we aren't going to be able to sync with the process state.
    if (current_mod_id.GetStopID() == 0)
    {
        m_exe_scope = m_target_sp.get();
        return false;
    }
        
    if (m_mod_id.IsValid())
    {
        if (m_mod_id == current_mod_id)
        {
            // Everything is already up to date in this object, no need do 
            // update the execution context scope.
            return false;
        }
        m_mod_id = current_mod_id;
        m_needs_update = true;        
    }
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
    // this will update the execution context scope and the m_mod_id
    SyncWithProcessState();
    m_first_update = false;
    m_needs_update = false;
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
            ProcessModID current_mod_id = process->GetModID();
            if (m_mod_id != current_mod_id)
            {
                needs_update = true;
                m_mod_id = current_mod_id;
            }
            // See if we're switching the thread or stack context.  If no thread is given, this is
            // being evaluated in a global context.            
            Thread *thread = exe_scope->CalculateThread();
            if (thread != NULL)
            {
                user_id_t new_thread_index = thread->GetIndexID();
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
