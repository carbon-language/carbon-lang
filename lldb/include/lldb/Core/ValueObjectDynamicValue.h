//===-- ValueObjectDynamicValue.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectDynamicValue_h_
#define liblldb_ValueObjectDynamicValue_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A ValueObject that represents memory at a given address, viewed as some 
// set lldb type.
//----------------------------------------------------------------------
class ValueObjectDynamicValue : public ValueObject
{
public:
    virtual
    ~ValueObjectDynamicValue();

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

    virtual bool
    IsInScope ();
    
    virtual bool
    IsDynamic ()
    {
        return true;
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

private:
    friend class ValueObject;
    ValueObjectDynamicValue (ValueObject &parent);

    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectDynamicValue);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectDynamicValue_h_
