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
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"

#include "lldb/Symbol/ClangASTType.h"
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

ValueObjectDynamicValue::ValueObjectDynamicValue (ValueObject &parent, lldb::DynamicValueType use_dynamic) :
    ValueObject(parent),
    m_address (),
    m_dynamic_type_info(),
    m_use_dynamic (use_dynamic)
{
    SetName (parent.GetName());
}

ValueObjectDynamicValue::~ValueObjectDynamicValue()
{
    m_owning_valobj_sp.reset();
}

ClangASTType
ValueObjectDynamicValue::GetClangTypeImpl ()
{
    const bool success = UpdateValueIfNeeded(false);
    if (success)
    {
        if (m_dynamic_type_info.HasType())
            return m_value.GetClangType();
        else
            return m_parent->GetClangType();
    }
    return m_parent->GetClangType();
}

ConstString
ValueObjectDynamicValue::GetTypeName()
{
    const bool success = UpdateValueIfNeeded(false);
    if (success)
    {
        if (m_dynamic_type_info.HasName())
            return m_dynamic_type_info.GetName();
    }
    return m_parent->GetTypeName();
}

TypeImpl
ValueObjectDynamicValue::GetTypeImpl ()
{
    const bool success = UpdateValueIfNeeded(false);
    if (success && m_type_impl.IsValid())
    {
        return m_type_impl;
    }
    return m_parent->GetTypeImpl();
}

ConstString
ValueObjectDynamicValue::GetQualifiedTypeName()
{
    const bool success = UpdateValueIfNeeded(false);
    if (success)
    {
        if (m_dynamic_type_info.HasName())
            return m_dynamic_type_info.GetName();
    }
    return m_parent->GetQualifiedTypeName();
}

ConstString
ValueObjectDynamicValue::GetDisplayTypeName()
{
    const bool success = UpdateValueIfNeeded(false);
    if (success)
    {
        if (m_dynamic_type_info.HasType())
            return GetClangType().GetDisplayTypeName();
        if (m_dynamic_type_info.HasName())
            return m_dynamic_type_info.GetName();
    }
    return m_parent->GetDisplayTypeName();
}

size_t
ValueObjectDynamicValue::CalculateNumChildren()
{
    const bool success = UpdateValueIfNeeded(false);
    if (success && m_dynamic_type_info.HasType())
        return GetClangType().GetNumChildren (true);
    else
        return m_parent->GetNumChildren();
}

uint64_t
ValueObjectDynamicValue::GetByteSize()
{
    const bool success = UpdateValueIfNeeded(false);
    if (success && m_dynamic_type_info.HasType())
        return m_value.GetValueByteSize(nullptr);
    else
        return m_parent->GetByteSize();
}

lldb::ValueType
ValueObjectDynamicValue::GetValueType() const
{
    return m_parent->GetValueType();
}


static TypeAndOrName
FixupTypeAndOrName (const TypeAndOrName& type_andor_name,
                    ValueObject& parent)
{
    TypeAndOrName ret(type_andor_name);
    if (type_andor_name.HasType())
    {
        // The type will always be the type of the dynamic object.  If our parent's type was a pointer,
        // then our type should be a pointer to the type of the dynamic object.  If a reference, then the original type
        // should be okay...
        ClangASTType orig_type = type_andor_name.GetClangASTType();
        ClangASTType corrected_type = orig_type;
        if (parent.IsPointerType())
            corrected_type = orig_type.GetPointerType ();
        else if (parent.IsPointerOrReferenceType())
            corrected_type = orig_type.GetLValueReferenceType ();
        ret.SetClangASTType(corrected_type);
    }
    else /*if (m_dynamic_type_info.HasName())*/
    {
        // If we are here we need to adjust our dynamic type name to include the correct & or * symbol
        std::string corrected_name (type_andor_name.GetName().GetCString());
        if (parent.IsPointerType())
            corrected_name.append(" *");
        else if (parent.IsPointerOrReferenceType())
            corrected_name.append(" &");
        // the parent type should be a correctly pointer'ed or referenc'ed type
        ret.SetClangASTType(parent.GetClangType());
        ret.SetName(corrected_name.c_str());
    }
    return ret;
}

bool
ValueObjectDynamicValue::UpdateValue ()
{
    SetValueIsValid (false);
    m_error.Clear();

    if (!m_parent->UpdateValueIfNeeded(false))
    {
        // The dynamic value failed to get an error, pass the error along
        if (m_error.Success() && m_parent->GetError().Fail())
            m_error = m_parent->GetError();
        return false;
    }

    // Setting our type_sp to NULL will route everything back through our
    // parent which is equivalent to not using dynamic values.
    if (m_use_dynamic == lldb::eNoDynamicValues)
    {
        m_dynamic_type_info.Clear();
        return true;
    }

    ExecutionContext exe_ctx (GetExecutionContextRef());
    Target *target = exe_ctx.GetTargetPtr();
    if (target)
    {
        m_data.SetByteOrder(target->GetArchitecture().GetByteOrder());
        m_data.SetAddressByteSize(target->GetArchitecture().GetAddressByteSize());
    }

    // First make sure our Type and/or Address haven't changed:
    Process *process = exe_ctx.GetProcessPtr();
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
            found_dynamic_type = runtime->GetDynamicTypeAndAddress (*m_parent, m_use_dynamic, class_type_or_name, dynamic_address);
    }
    else
    {
        LanguageRuntime *cpp_runtime = process->GetLanguageRuntime (lldb::eLanguageTypeC_plus_plus);
        if (cpp_runtime)
            found_dynamic_type = cpp_runtime->GetDynamicTypeAndAddress (*m_parent, m_use_dynamic, class_type_or_name, dynamic_address);

        if (!found_dynamic_type)
        {
            LanguageRuntime *objc_runtime = process->GetLanguageRuntime (lldb::eLanguageTypeObjC);
            if (objc_runtime)
                found_dynamic_type = objc_runtime->GetDynamicTypeAndAddress (*m_parent, m_use_dynamic, class_type_or_name, dynamic_address);
        }
    }

    // Getting the dynamic value may have run the program a bit, and so marked us as needing updating, but we really
    // don't...

    m_update_point.SetUpdated();

    if (found_dynamic_type)
    {
        if (class_type_or_name.HasType())
        {
            m_type_impl = TypeImpl(m_parent->GetClangType(),FixupTypeAndOrName(class_type_or_name, *m_parent).GetClangASTType());
        }
        else
        {
            m_type_impl.Clear();
        }
    }
    else
    {
        m_type_impl.Clear();
    }

    // If we don't have a dynamic type, then make ourselves just a echo of our parent.
    // Or we could return false, and make ourselves an echo of our parent?
    if (!found_dynamic_type)
    {
        if (m_dynamic_type_info)
            SetValueDidChange(true);
        ClearDynamicTypeInformation();
        m_dynamic_type_info.Clear();
        m_value = m_parent->GetValue();
        m_error = m_value.GetValueAsData (&exe_ctx, m_data, 0, GetModule().get());
        return m_error.Success();
    }

    Value old_value(m_value);

    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));

    bool has_changed_type = false;

    if (!m_dynamic_type_info)
    {
        m_dynamic_type_info = class_type_or_name;
        has_changed_type = true;
    }
    else if (class_type_or_name != m_dynamic_type_info)
    {
        // We are another type, we need to tear down our children...
        m_dynamic_type_info = class_type_or_name;
        SetValueDidChange (true);
        has_changed_type = true;
    }

    if (has_changed_type)
        ClearDynamicTypeInformation ();

    if (!m_address.IsValid() || m_address != dynamic_address)
    {
        if (m_address.IsValid())
            SetValueDidChange (true);

        // We've moved, so we should be fine...
        m_address = dynamic_address;
        lldb::TargetSP target_sp (GetTargetSP());
        lldb::addr_t load_address = m_address.GetLoadAddress(target_sp.get());
        m_value.GetScalar() = load_address;
    }

    m_dynamic_type_info = FixupTypeAndOrName(m_dynamic_type_info, *m_parent);

    //m_value.SetContext (Value::eContextTypeClangType, corrected_type);
    m_value.SetClangType (m_dynamic_type_info.GetClangASTType());

    // Our address is the location of the dynamic type stored in memory.  It isn't a load address,
    // because we aren't pointing to the LOCATION that stores the pointer to us, we're pointing to us...
    m_value.SetValueType(Value::eValueTypeScalar);

    if (has_changed_type && log)
        log->Printf("[%s %p] has a new dynamic type %s", GetName().GetCString(),
                    static_cast<void*>(this), GetTypeName().GetCString());

    if (m_address.IsValid() && m_dynamic_type_info)
    {
        // The variable value is in the Scalar value inside the m_value.
        // We can point our m_data right to it.
        m_error = m_value.GetValueAsData (&exe_ctx, m_data, 0, GetModule().get());
        if (m_error.Success())
        {
            if (!CanProvideValue())
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

bool
ValueObjectDynamicValue::SetValueFromCString (const char *value_str, Error& error)
{
    if (!UpdateValueIfNeeded(false))
    {
        error.SetErrorString("unable to read value");
        return false;
    }
    
    uint64_t my_value = GetValueAsUnsigned(UINT64_MAX);
    uint64_t parent_value = m_parent->GetValueAsUnsigned(UINT64_MAX);
    
    if (my_value == UINT64_MAX || parent_value == UINT64_MAX)
    {
        error.SetErrorString("unable to read value");
        return false;
    }
    
    // if we are at an offset from our parent, in order to set ourselves correctly we would need
    // to change the new value so that it refers to the correct dynamic type. we choose not to deal
    // with that - if anything more than a value overwrite is required, you should be using the
    // expression parser instead of the value editing facility
    if (my_value != parent_value)
    {
        // but NULL'ing out a value should always be allowed
        if (strcmp(value_str,"0"))
        {
            error.SetErrorString("unable to modify dynamic value, use 'expression' command");
            return false;
        }
    }
    
    bool ret_val = m_parent->SetValueFromCString(value_str,error);
    SetNeedsUpdate();
    return ret_val;
}

bool
ValueObjectDynamicValue::SetData (DataExtractor &data, Error &error)
{
    if (!UpdateValueIfNeeded(false))
    {
        error.SetErrorString("unable to read value");
        return false;
    }
    
    uint64_t my_value = GetValueAsUnsigned(UINT64_MAX);
    uint64_t parent_value = m_parent->GetValueAsUnsigned(UINT64_MAX);
    
    if (my_value == UINT64_MAX || parent_value == UINT64_MAX)
    {
        error.SetErrorString("unable to read value");
        return false;
    }
    
    // if we are at an offset from our parent, in order to set ourselves correctly we would need
    // to change the new value so that it refers to the correct dynamic type. we choose not to deal
    // with that - if anything more than a value overwrite is required, you should be using the
    // expression parser instead of the value editing facility
    if (my_value != parent_value)
    {
        // but NULL'ing out a value should always be allowed
        lldb::offset_t offset = 0;
        
        if (data.GetPointer(&offset) != 0)
        {
            error.SetErrorString("unable to modify dynamic value, use 'expression' command");
            return false;
        }
    }
    
    bool ret_val = m_parent->SetData(data, error);
    SetNeedsUpdate();
    return ret_val;
}
