//===-- ValueObject.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObject_h_
#define liblldb_ValueObject_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/Value.h"
#include "lldb/Target/ExecutionContextScope.h"

namespace lldb_private {

class ValueObject : public UserID
{
public:
    friend class ValueObjectList;

    virtual ~ValueObject();

    //------------------------------------------------------------------
    // Sublasses must implement the functions below.
    //------------------------------------------------------------------
    virtual size_t
    GetByteSize() = 0;

    virtual clang::ASTContext *
    GetClangAST () = 0;

    virtual void *
    GetClangType () = 0;

    virtual lldb::ValueType
    GetValueType() const = 0;

protected:
    // Should only be called by ValueObject::GetNumChildren()
    virtual uint32_t
    CalculateNumChildren() = 0;

public:
    virtual ConstString
    GetTypeName() = 0;

    virtual lldb::LanguageType
    GetObjectRuntimeLanguage();
    
    virtual void
    UpdateValue (ExecutionContextScope *exe_scope) = 0;

    //------------------------------------------------------------------
    // Sublasses can implement the functions below if they need to.
    //------------------------------------------------------------------
protected:
    // Should only be called by ValueObject::GetChildAtIndex()
    virtual lldb::ValueObjectSP
    CreateChildAtIndex (uint32_t idx, bool synthetic_array_member, int32_t synthetic_index);

public:

    virtual bool
    IsPointerType ();

    virtual bool
    IsPointerOrReferenceType ();

    virtual bool
    IsBaseClass ()
    {
        return false;
    }

    virtual void
    GetExpressionPath (Stream &s);

    virtual bool
    IsInScope (StackFrame *frame)
    {
        return true;
    }

    virtual off_t
    GetByteOffset()
    {
        return 0;
    }

    virtual uint32_t
    GetBitfieldBitSize()
    {
        return 0;
    }

    virtual uint32_t
    GetBitfieldBitOffset()
    {
        return 0;
    }
    
    virtual bool
    SetClangAST (clang::ASTContext *ast)
    {
        return false;
    }

    virtual const char *
    GetValueAsCString (ExecutionContextScope *exe_scope);

    virtual bool
    SetValueFromCString (ExecutionContextScope *exe_scope, const char *value_str);

    //------------------------------------------------------------------
    // The functions below should NOT be modified by sublasses
    //------------------------------------------------------------------
    const Error &
    GetError() const;

    const ConstString &
    GetName() const;

    lldb::ValueObjectSP
    GetChildAtIndex (uint32_t idx, bool can_create);

    lldb::ValueObjectSP
    GetChildMemberWithName (const ConstString &name, bool can_create);

    uint32_t
    GetIndexOfChildWithName (const ConstString &name);

    uint32_t
    GetNumChildren ();

    const Value &
    GetValue() const;

    Value &
    GetValue();

    bool
    ResolveValue (ExecutionContextScope *exe_scope, Scalar &scalar);

    const char *
    GetLocationAsCString (ExecutionContextScope *exe_scope);

    const char *
    GetSummaryAsCString (ExecutionContextScope *exe_scope);
    
    const char *
    GetObjectDescription (ExecutionContextScope *exe_scope);


    lldb::user_id_t
    GetUpdateID() const;

    bool
    GetValueIsValid () const;

    bool
    GetValueDidChange (ExecutionContextScope *exe_scope);

    bool
    UpdateValueIfNeeded (ExecutionContextScope *exe_scope);

    const DataExtractor &
    GetDataExtractor () const;

    DataExtractor &
    GetDataExtractor ();

    bool
    Write ();

    void
    AddSyntheticChild (const ConstString &key,
                       lldb::ValueObjectSP& valobj_sp);

    lldb::ValueObjectSP
    GetSyntheticChild (const ConstString &key) const;

    lldb::ValueObjectSP
    GetSyntheticArrayMemberFromPointer (int32_t index, bool can_create);
    
    lldb::ValueObjectSP
    GetDynamicValue ()
    {
        return m_dynamic_value_sp;
    }
    
    virtual lldb::ValueObjectSP
    CreateConstantValue (ExecutionContextScope *exe_scope, const ConstString &name);

    virtual lldb::ValueObjectSP
    Dereference (Error &error);
    
    virtual lldb::ValueObjectSP
    AddressOf (Error &error);

    // The backing bits of this value object were updated, clear any value
    // values, summaries or descriptions so we refetch them.
    virtual void
    ValueUpdated ()
    {
        m_value_str.clear();
        m_summary_str.clear();
        m_object_desc_str.clear();
    }

    bool
    SetDynamicValue ();

    static void
    DumpValueObject (Stream &s,
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
                     bool flat_output);

    bool
    GetIsConstant () const
    {
        return m_update_id == LLDB_INVALID_UID;
    }
    
    void
    SetIsConstant ()
    {
        m_update_id = LLDB_INVALID_UID;
    }

    lldb::Format
    GetFormat () const
    {
        return m_format;
    }
    
    void
    SetFormat (lldb::Format format)
    {
        if (format != m_format)
            m_value_str.clear();
        m_format = format;
    }

    ValueObject *
    GetParent()
    {
        return m_parent;
    }

    const ValueObject *
    GetParent() const
    {
        return m_parent;
    }
    
    void
    SetPointersPointToLoadAddrs (bool b)
    {
        m_pointers_point_to_load_addrs = b;
    }

protected:
    //------------------------------------------------------------------
    // Classes that inherit from ValueObject can see and modify these
    //------------------------------------------------------------------
    ValueObject*        m_parent;       // The parent value object, or NULL if this has no parent
    lldb::user_id_t     m_update_id;    // An integer that specifies the update number for this value in
                                        // this value object list. If this value object is asked to update itself
                                        // it will first check if the update ID match the value object
                                        // list update number. If the update numbers match, no update is
                                        // needed, if it does not match, this value object should update its
                                        // the next time it is asked.
    ConstString         m_name;         // The name of this object
    DataExtractor       m_data;         // A data extractor that can be used to extract the value.
    Value               m_value;
    Error               m_error;        // An error object that can describe any errors that occur when updating values.
    std::string         m_value_str;    // Cached value string that will get cleared if/when the value is updated.
    std::string         m_old_value_str;// Cached old value string from the last time the value was gotten
    std::string         m_location_str; // Cached location string that will get cleared if/when the value is updated.
    std::string         m_summary_str;  // Cached summary string that will get cleared if/when the value is updated.
    std::string         m_object_desc_str; // Cached result of the "object printer".  This differs from the summary
                                              // in that the summary is consed up by us, the object_desc_string is builtin.
    std::vector<lldb::ValueObjectSP> m_children;
    std::map<ConstString, lldb::ValueObjectSP> m_synthetic_children;
    lldb::ValueObjectSP m_dynamic_value_sp;
    lldb::Format        m_format;
    bool                m_value_is_valid:1,
                        m_value_did_change:1,
                        m_children_count_valid:1,
                        m_old_value_valid:1,
                        m_pointers_point_to_load_addrs:1;
    
    friend class CommandObjectExpression;
    friend class ClangExpressionVariable;
    friend class Target;
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ValueObject (ValueObject *parent);

    void
    SetName (const char *name);

    void
    SetName (const ConstString &name);

    void
    SetNumChildren (uint32_t num_children);

    void
    SetValueDidChange (bool value_changed);

    void
    SetValueIsValid (bool valid);


    lldb::addr_t
    GetPointerValue (lldb::AddressType &address_type, 
                     bool scalar_is_load_address);

    lldb::addr_t
    GetAddressOf (lldb::AddressType &address_type, 
                  bool scalar_is_load_address);
private:
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObject);

};

} // namespace lldb_private

#endif  // liblldb_ValueObject_h_
