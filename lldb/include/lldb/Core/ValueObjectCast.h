//===-- ValueObjectDynamicValue.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectCast_h_
#define liblldb_ValueObjectCast_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//---------------------------------------------------------------------------------
// A ValueObject that represents a given value represented as a different type.
//---------------------------------------------------------------------------------
class ValueObjectCast : public ValueObject
{
public:
    static lldb::ValueObjectSP
    Create (ValueObject &parent, 
            const ConstString &name, 
            const CompilerType &cast_type);

    ~ValueObjectCast() override;
    
    uint64_t
    GetByteSize() override;
    
    size_t
    CalculateNumChildren() override;
    
    lldb::ValueType
    GetValueType() const override;
    
    bool
    IsInScope() override;
    
    ValueObject *
    GetParent() override
    {
        if (m_parent)
            return m_parent->GetParent();
        else
            return NULL;
    }
    
    const ValueObject *
    GetParent() const override
    {
        if (m_parent)
            return m_parent->GetParent();
        else
            return NULL;
    }
    
protected:
    bool
    UpdateValue () override;
    
    CompilerType
    GetCompilerTypeImpl () override;
    
    CompilerType m_cast_type;
    
    ValueObjectCast (ValueObject &parent, 
                     const ConstString &name, 
                     const CompilerType &cast_type);
    
private:
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectCast);
};

} // namespace lldb_private

#endif // liblldb_ValueObjectCast_h_
