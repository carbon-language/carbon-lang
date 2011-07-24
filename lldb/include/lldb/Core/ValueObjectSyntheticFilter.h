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
#include <map>
#include <ostream>
#include <vector>
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A ValueObject that obtains its children from some source other than
// real information
// This is currently used to implement children filtering, where only
// a subset of the real children are shown, but it can be used for any
// source of made-up children information
//----------------------------------------------------------------------
class ValueObjectSynthetic : public ValueObject
{
public:
    virtual
    ~ValueObjectSynthetic();

    virtual size_t
    GetByteSize();

    virtual clang::ASTContext *
    GetClangAST ();

    virtual lldb::clang_type_t
    GetClangType ();

    virtual ConstString
    GetTypeName();

    virtual uint32_t
    CalculateNumChildren();

    virtual lldb::ValueType
    GetValueType() const;
    
    virtual lldb::ValueObjectSP
    GetChildAtIndex (uint32_t idx, bool can_create);
    
    virtual lldb::ValueObjectSP
    GetChildMemberWithName (const ConstString &name, bool can_create);
    
    virtual uint32_t
    GetIndexOfChildWithName (const ConstString &name);

    virtual bool
    IsInScope ();
    
    virtual bool
    IsDynamic ()
    {
        if (m_parent)
            return m_parent->IsDynamic();
        else
            return false;
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

    void
    SetOwningSP (lldb::ValueObjectSP &owning_sp)
    {
        if (m_owning_valobj_sp == owning_sp)
            return;
            
        assert (m_owning_valobj_sp.get() == NULL);
        m_owning_valobj_sp = owning_sp;
    }
    
protected:
    virtual bool
    UpdateValue ();

    Address  m_address;  ///< The variable that this value object is based upon
    lldb::TypeSP m_type_sp;
    lldb::ValueObjectSP m_owning_valobj_sp;
    lldb::SyntheticValueType m_use_synthetic;
    lldb::SyntheticChildrenFrontEndSP m_synth_filter;
    
    typedef std::map<uint32_t, lldb::ValueObjectSP> ByIndexMap;
    typedef std::map<const char*, uint32_t> NameToIndexMap;
    
    typedef ByIndexMap::iterator ByIndexIterator;
    typedef NameToIndexMap::iterator NameToIndexIterator;
    
    ByIndexMap m_children_byindex;
    NameToIndexMap m_name_toindex;

private:
    friend class ValueObject;
    ValueObjectSynthetic (ValueObject &parent, lldb::SyntheticChildrenSP filter);
    
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectSynthetic);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectSyntheticFilter_h_
