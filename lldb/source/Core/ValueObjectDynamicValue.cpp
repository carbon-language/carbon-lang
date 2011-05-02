//===-- ValueObjectDynamicValue.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/Core/ValueObjectDynamicValue.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Module.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"

#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"


using namespace lldb_private;

ValueObjectDynamicValue::ValueObjectDynamicValue (ValueObject &parent) :
    ValueObject(parent),
    m_address (),
    m_type_sp()
{
    SetName (parent.GetName().AsCString());
}

ValueObjectDynamicValue::~ValueObjectDynamicValue()
{
    m_owning_valobj_sp.reset();
}

lldb::clang_type_t
ValueObjectDynamicValue::GetClangType ()
{
    if (m_type_sp)
        return m_value.GetClangType();
    else
        return m_parent->GetClangType();
}

ConstString
ValueObjectDynamicValue::GetTypeName()
{
    // FIXME: Maybe cache the name, but have to clear it out if the type changes...
    if (!UpdateValueIfNeeded())
        return ConstString("<unknown type>");
        
    if (m_type_sp)
        return ClangASTType::GetClangTypeName (GetClangType());
    else
        return m_parent->GetTypeName();
}

uint32_t
ValueObjectDynamicValue::CalculateNumChildren()
{
    if (!UpdateValueIfNeeded())
        return 0;
        
    if (m_type_sp)
        return ClangASTContext::GetNumChildren (GetClangAST (), GetClangType(), true);
    else
        return m_parent->GetNumChildren();
}

clang::ASTContext *
ValueObjectDynamicValue::GetClangAST ()
{
    if (!UpdateValueIfNeeded())
        return NULL;
        
    if (m_type_sp)
        return m_type_sp->GetClangAST();
    else
        return m_parent->GetClangAST ();
}

size_t
ValueObjectDynamicValue::GetByteSize()
{
    if (!UpdateValueIfNeeded())
        return 0;
        
    if (m_type_sp)
        return m_value.GetValueByteSize(GetClangAST(), NULL);
    else
        return m_parent->GetByteSize();
}

lldb::ValueType
ValueObjectDynamicValue::GetValueType() const
{
    return m_parent->GetValueType();
}

bool
ValueObjectDynamicValue::UpdateValue ()
{
    SetValueIsValid (false);
    m_error.Clear();

    if (!m_parent->UpdateValueIfNeeded())
    {
        return false;
    }
    
    ExecutionContext exe_ctx (GetExecutionContextScope());
    
    if (exe_ctx.target)
    {
        m_data.SetByteOrder(exe_ctx.target->GetArchitecture().GetByteOrder());
        m_data.SetAddressByteSize(exe_ctx.target->GetArchitecture().GetAddressByteSize());
    }
    
    // First make sure our Type and/or Address haven't changed:
    Process *process = m_update_point.GetProcess();
    if (!process)
        return false;
    
    TypeAndOrName class_type_or_name;
    Address dynamic_address;
    bool found_dynamic_type = false;
    
    lldb::LanguageType known_type = m_parent->GetObjectRuntimeLanguage();
    if (known_type != lldb::eLanguageTypeUnknown && known_type != lldb::eLanguageTypeC)
    {
        LanguageRuntime *runtime = process->GetLanguageRuntime (known_type);
        if (runtime)
            found_dynamic_type = runtime->GetDynamicTypeAndAddress (*m_parent, class_type_or_name, dynamic_address);
    }
    else
    {
        LanguageRuntime *cpp_runtime = process->GetLanguageRuntime (lldb::eLanguageTypeC_plus_plus);
        if (cpp_runtime)
            found_dynamic_type = cpp_runtime->GetDynamicTypeAndAddress (*m_parent, class_type_or_name, dynamic_address);
        
        if (!found_dynamic_type)
        {
            LanguageRuntime *objc_runtime = process->GetLanguageRuntime (lldb::eLanguageTypeObjC);
            if (objc_runtime)
                found_dynamic_type = cpp_runtime->GetDynamicTypeAndAddress (*m_parent, class_type_or_name, dynamic_address);
        }
    }
    
    lldb::TypeSP dynamic_type_sp = class_type_or_name.GetTypeSP();
    
    // Getting the dynamic value may have run the program a bit, and so marked us as needing updating, but we really
    // don't...
    
    m_update_point.SetUpdated();
    
    // If we don't have a dynamic type, then make ourselves just a echo of our parent.
    // Or we could return false, and make ourselves an echo of our parent?
    if (!found_dynamic_type)
    {
        if (m_type_sp)
            SetValueDidChange(true);
        m_value = m_parent->GetValue();
        m_error = m_value.GetValueAsData (&exe_ctx, GetClangAST(), m_data, 0);
        return m_error.Success();
    }
    
    Value old_value(m_value);

    if (!m_type_sp)
    {
        m_type_sp = dynamic_type_sp;
    }
    else if (dynamic_type_sp != m_type_sp)
    {
        // We are another type, we need to tear down our children...
        m_type_sp = dynamic_type_sp;
        SetValueDidChange (true);
    }
    
    if (!m_address.IsValid() || m_address != dynamic_address)
    {
        if (m_address.IsValid())
            SetValueDidChange (true);
            
        // We've moved, so we should be fine...
        m_address = dynamic_address;
        lldb::addr_t load_address = m_address.GetLoadAddress(m_update_point.GetTarget());
        m_value.GetScalar() = load_address;
    }
    
    // The type will always be the type of the dynamic object.  If our parent's type was a pointer,
    // then our type should be a pointer to the type of the dynamic object.  If a reference, then the original type
    // should be okay...
    lldb::clang_type_t orig_type = m_type_sp->GetClangForwardType();
    lldb::clang_type_t corrected_type = orig_type;
    if (m_parent->IsPointerType())
        corrected_type = ClangASTContext::CreatePointerType (m_type_sp->GetClangAST(), orig_type);
    else if (m_parent->IsPointerOrReferenceType())
        corrected_type = ClangASTContext::CreateLValueReferenceType (m_type_sp->GetClangAST(), orig_type);
        
    m_value.SetContext (Value::eContextTypeClangType, corrected_type);
    
    // Our address is the location of the dynamic type stored in memory.  It isn't a load address,
    // because we aren't pointing to the LOCATION that stores the pointer to us, we're pointing to us...
    m_value.SetValueType(Value::eValueTypeScalar);

    if (m_address.IsValid() && m_type_sp)
    {
        // The variable value is in the Scalar value inside the m_value.
        // We can point our m_data right to it.
        m_error = m_value.GetValueAsData (&exe_ctx, GetClangAST(), m_data, 0);
        if (m_error.Success())
        {
            if (ClangASTContext::IsAggregateType (GetClangType()))
            {
                // this value object represents an aggregate type whose
                // children have values, but this object does not. So we
                // say we are changed if our location has changed.
                SetValueDidChange (m_value.GetValueType() != old_value.GetValueType() || m_value.GetScalar() != old_value.GetScalar());
            }

            SetValueIsValid (true);
            return true;
        }
    }
    
    // We get here if we've failed above...
    SetValueIsValid (false);
    return false;
}



bool
ValueObjectDynamicValue::IsInScope ()
{
    return m_parent->IsInScope();
}

