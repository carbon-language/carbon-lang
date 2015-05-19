//===-- ValueObjectSyntheticFilter.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectSyntheticFilter_h_
#define liblldb_ValueObjectSyntheticFilter_h_

// C Includes
// C++ Includes
#include <vector>
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ThreadSafeSTLMap.h"
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A ValueObject that obtains its children from some source other than
// real information
// This is currently used to implement Python-based children and filters
// but you can bind it to any source of synthetic information and have
// it behave accordingly
//----------------------------------------------------------------------
class ValueObjectSynthetic : public ValueObject
{
public:
    virtual
    ~ValueObjectSynthetic();

    virtual uint64_t
    GetByteSize();
    
    virtual ConstString
    GetTypeName();
    
    virtual ConstString
    GetQualifiedTypeName();
    
    virtual ConstString
    GetDisplayTypeName();

    virtual bool
    MightHaveChildren();

    virtual size_t
    CalculateNumChildren();

    virtual lldb::ValueType
    GetValueType() const;
    
    virtual lldb::ValueObjectSP
    GetChildAtIndex (size_t idx, bool can_create);
    
    virtual lldb::ValueObjectSP
    GetChildMemberWithName (const ConstString &name, bool can_create);
    
    virtual size_t
    GetIndexOfChildWithName (const ConstString &name);

    virtual lldb::ValueObjectSP
    GetDynamicValue (lldb::DynamicValueType valueType);
    
    virtual bool
    IsInScope ();
    
    virtual bool
    HasSyntheticValue()
    {
        return false;
    }
    
    virtual bool
    IsSynthetic() { return true; }
    
    virtual void
    CalculateSyntheticValue (bool use_synthetic)
    {
    }
    
    virtual bool
    IsDynamic ()
    {
        if (m_parent)
            return m_parent->IsDynamic();
        else
            return false;
    }
    
    virtual lldb::ValueObjectSP
    GetStaticValue ()
    {
        if (m_parent)
            return m_parent->GetStaticValue();
        else
            return GetSP();
    }
    
    virtual lldb::DynamicValueType
    GetDynamicValueType ()
    {
        if (m_parent)
            return m_parent->GetDynamicValueType();
        else
            return lldb::eNoDynamicValues;
    }

    virtual ValueObject *
    GetParent()
    {
        if (m_parent)
            return m_parent->GetParent();
        else
            return NULL;
    }

    virtual const ValueObject *
    GetParent() const
    {
        if (m_parent)
            return m_parent->GetParent();
        else
            return NULL;
    }
    
    virtual lldb::ValueObjectSP
    GetNonSyntheticValue ();
    
    virtual bool
    CanProvideValue ();
    
    virtual bool
    DoesProvideSyntheticValue ()
    {
        return (UpdateValueIfNeeded(), m_provides_value == eLazyBoolYes);
    }
    
    virtual bool
    GetIsConstant () const
    {
        return false;
    }

    virtual bool
    SetValueFromCString (const char *value_str, Error& error);
    
    virtual void
    SetFormat (lldb::Format format);
    
protected:
    virtual bool
    UpdateValue ();
    
    virtual bool
    CanUpdateWithInvalidExecutionContext ()
    {
        return true;
    }
    
    virtual ClangASTType
    GetClangTypeImpl ();
    
    virtual void
    CreateSynthFilter ();

    // we need to hold on to the SyntheticChildren because someone might delete the type binding while we are alive
    lldb::SyntheticChildrenSP m_synth_sp;
    std::unique_ptr<SyntheticChildrenFrontEnd> m_synth_filter_ap;
    
    typedef ThreadSafeSTLMap<uint32_t, ValueObject*> ByIndexMap;
    typedef ThreadSafeSTLMap<const char*, uint32_t> NameToIndexMap;
    
    typedef ByIndexMap::iterator ByIndexIterator;
    typedef NameToIndexMap::iterator NameToIndexIterator;

    ByIndexMap      m_children_byindex;
    NameToIndexMap  m_name_toindex;
    uint32_t        m_synthetic_children_count; // FIXME use the ValueObject's ChildrenManager instead of a special purpose solution
    
    ConstString     m_parent_type_name;

    LazyBool        m_might_have_children;
    
    LazyBool        m_provides_value;
    
private:
    friend class ValueObject;
    ValueObjectSynthetic (ValueObject &parent, lldb::SyntheticChildrenSP filter);
    
    void
    CopyValueData (ValueObject *source);
    
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectSynthetic);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectSyntheticFilter_h_
