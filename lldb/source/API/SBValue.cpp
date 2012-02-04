//===-- SBValue.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBValue.h"
#include "lldb/API/SBStream.h"

#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBDebugger.h"

using namespace lldb;
using namespace lldb_private;

SBValue::SBValue () :
    m_opaque_sp ()
{
}

SBValue::SBValue (const lldb::ValueObjectSP &value_sp) :
    m_opaque_sp (value_sp)
{
}

SBValue::SBValue(const SBValue &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

SBValue &
SBValue::operator = (const SBValue &rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
}

SBValue::~SBValue()
{
}

bool
SBValue::IsValid ()
{
    // If this function ever changes to anything that does more than just
    // check if the opaque shared pointer is non NULL, then we need to update
    // all "if (m_opaque_sp)" code in this file.
    return m_opaque_sp.get() != NULL;
}

void
SBValue::Clear()
{
    m_opaque_sp.reset();
}

SBError
SBValue::GetError()
{
    SBError sb_error;
    
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        sb_error.SetError(value_sp->GetError());
    else
        sb_error.SetErrorString("error: invalid value");
    
    return sb_error;
}

user_id_t
SBValue::GetID()
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        return value_sp->GetID();
    return LLDB_INVALID_UID;
}

const char *
SBValue::GetName()
{

    const char *name = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        name = value_sp->GetName().GetCString();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (name)
            log->Printf ("SBValue(%p)::GetName () => \"%s\"", value_sp.get(), name);
        else
            log->Printf ("SBValue(%p)::GetName () => NULL", value_sp.get());
    }

    return name;
}

const char *
SBValue::GetTypeName ()
{
    const char *name = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        name = value_sp->GetTypeName().GetCString();
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (name)
            log->Printf ("SBValue(%p)::GetTypeName () => \"%s\"", value_sp.get(), name);
        else
            log->Printf ("SBValue(%p)::GetTypeName () => NULL", value_sp.get());
    }

    return name;
}

size_t
SBValue::GetByteSize ()
{
    size_t result = 0;

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        result = value_sp->GetByteSize();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetByteSize () => %zu", value_sp.get(), result);

    return result;
}

bool
SBValue::IsInScope ()
{
    bool result = false;

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            result = value_sp->IsInScope ();
        }
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::IsInScope () => %i", value_sp.get(), result);

    return result;
}

const char *
SBValue::GetValue ()
{
    const char *cstr = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            cstr = value_sp->GetValueAsCString ();
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetValue => \"%s\"", value_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetValue => NULL", value_sp.get());
    }

    return cstr;
}

ValueType
SBValue::GetValueType ()
{
    ValueType result = eValueTypeInvalid;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        result = value_sp->GetValueType();
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        switch (result)
        {
        case eValueTypeInvalid:         log->Printf ("SBValue(%p)::GetValueType () => eValueTypeInvalid", value_sp.get()); break;
        case eValueTypeVariableGlobal:  log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableGlobal", value_sp.get()); break;
        case eValueTypeVariableStatic:  log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableStatic", value_sp.get()); break;
        case eValueTypeVariableArgument:log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableArgument", value_sp.get()); break;
        case eValueTypeVariableLocal:   log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableLocal", value_sp.get()); break;
        case eValueTypeRegister:        log->Printf ("SBValue(%p)::GetValueType () => eValueTypeRegister", value_sp.get()); break;
        case eValueTypeRegisterSet:     log->Printf ("SBValue(%p)::GetValueType () => eValueTypeRegisterSet", value_sp.get()); break;
        case eValueTypeConstResult:     log->Printf ("SBValue(%p)::GetValueType () => eValueTypeConstResult", value_sp.get()); break;
        default:     log->Printf ("SBValue(%p)::GetValueType () => %i ???", value_sp.get(), result); break;
        }
    }
    return result;
}

const char *
SBValue::GetObjectDescription ()
{
    const char *cstr = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            cstr = value_sp->GetObjectDescription ();
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetObjectDescription => \"%s\"", value_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetObjectDescription => NULL", value_sp.get());
    }
    return cstr;
}

SBType
SBValue::GetType()
{
    SBType sb_type;
    lldb::ValueObjectSP value_sp(GetSP());
    TypeImplSP type_sp;
    if (value_sp)
    {
        type_sp.reset (new TypeImpl(ClangASTType (value_sp->GetClangAST(), value_sp->GetClangType())));
        sb_type.SetSP(type_sp);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (type_sp)
            log->Printf ("SBValue(%p)::GetType => SBType(%p)", value_sp.get(), type_sp.get());
        else
            log->Printf ("SBValue(%p)::GetType => NULL", value_sp.get());
    }
    return sb_type;
}

bool
SBValue::GetValueDidChange ()
{
    bool result = false;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            result = value_sp->GetValueDidChange ();
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetValueDidChange => %i", value_sp.get(), result);

    return result;
}

const char *
SBValue::GetSummary ()
{
    const char *cstr = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            cstr = value_sp->GetSummaryAsCString();
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetSummary => \"%s\"", value_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetSummary => NULL", value_sp.get());
    }
    return cstr;
}

const char *
SBValue::GetLocation ()
{
    const char *cstr = NULL;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            cstr = value_sp->GetLocationAsCString();
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetSummary => \"%s\"", value_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetSummary => NULL", value_sp.get());
    }
    return cstr;
}

bool
SBValue::SetValueFromCString (const char *value_str)
{
    bool success = false;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            success = value_sp->SetValueFromCString (value_str);
        }
    }
    return success;
}

lldb::SBValue
SBValue::CreateChildAtOffset (const char *name, uint32_t offset, SBType type)
{
    lldb::SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    lldb::ValueObjectSP new_value_sp;
    if (value_sp)
    {
        TypeImplSP type_sp (type.GetSP());
        if (type.IsValid())
        {
            sb_value = SBValue(value_sp->GetSyntheticChildAtOffset(offset, type_sp->GetClangASTType(), true));
            new_value_sp = sb_value.GetSP();
            if (new_value_sp)
                new_value_sp->SetName(ConstString(name));
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (new_value_sp)
            log->Printf ("SBValue(%p)::GetChildAtOffset => \"%s\"", value_sp.get(), new_value_sp->GetName().AsCString());
        else
            log->Printf ("SBValue(%p)::GetChildAtOffset => NULL", value_sp.get());
    }
    return sb_value;
}

lldb::SBValue
SBValue::Cast (SBType type)
{
    lldb::SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    TypeImplSP type_sp (type.GetSP());
    if (value_sp && type_sp)
        sb_value.SetSP(value_sp->Cast(type_sp->GetClangASTType()));
    return sb_value;
}

lldb::SBValue
SBValue::CreateValueFromExpression (const char *name, const char* expression)
{
    lldb::SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    lldb::ValueObjectSP new_value_sp;
    if (value_sp)
    {
        value_sp->GetUpdatePoint().GetTargetSP()->EvaluateExpression (expression,
                                                                      value_sp->GetExecutionContextScope()->CalculateStackFrame(),
                                                                      eExecutionPolicyOnlyWhenNeeded,
                                                                      false, // coerce to id
                                                                      true, // unwind on error
                                                                      true, // keep in memory
                                                                      eNoDynamicValues,
                                                                      new_value_sp);
        if (new_value_sp)
        {
            new_value_sp->SetName(ConstString(name));
            sb_value.SetSP(new_value_sp);
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (new_value_sp)
            log->Printf ("SBValue(%p)::GetChildFromExpression => \"%s\"", value_sp.get(), new_value_sp->GetName().AsCString());
        else
            log->Printf ("SBValue(%p)::GetChildFromExpression => NULL", value_sp.get());
    }
    return sb_value;
}

lldb::SBValue
SBValue::CreateValueFromAddress(const char* name, lldb::addr_t address, SBType sb_type)
{
    lldb::SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    lldb::ValueObjectSP new_value_sp;
    lldb::TypeImplSP type_impl_sp (sb_type.GetSP());
    if (value_sp && type_impl_sp)
    {
        ClangASTType pointee_ast_type(type_impl_sp->GetASTContext(), type_impl_sp->GetClangASTType().GetPointerType ());
        lldb::TypeImplSP pointee_type_impl_sp (new TypeImpl(pointee_ast_type));
        if (pointee_type_impl_sp)
        {
        
            lldb::DataBufferSP buffer(new lldb_private::DataBufferHeap(&address,sizeof(lldb::addr_t)));
        
            ValueObjectSP ptr_result_valobj_sp(ValueObjectConstResult::Create (value_sp->GetExecutionContextScope(),
                                                                               pointee_type_impl_sp->GetASTContext(),
                                                                               pointee_type_impl_sp->GetOpaqueQualType(),
                                                                               ConstString(name),
                                                                               buffer,
                                                                               lldb::endian::InlHostByteOrder(), 
                                                                               GetTarget().GetProcess().GetAddressByteSize()));
        
            if (ptr_result_valobj_sp)
            {
                ptr_result_valobj_sp->GetValue().SetValueType(Value::eValueTypeLoadAddress);
                Error err;
                new_value_sp = ptr_result_valobj_sp->Dereference(err);
                if (new_value_sp)
                    new_value_sp->SetName(ConstString(name));
            }
            sb_value.SetSP(new_value_sp);
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (new_value_sp)
            log->Printf ("SBValue(%p)::CreateValueFromAddress => \"%s\"", value_sp.get(), new_value_sp->GetName().AsCString());
        else
            log->Printf ("SBValue(%p)::CreateValueFromAddress => NULL", value_sp.get());
    }
    return sb_value;
}

lldb::SBValue
SBValue::CreateValueFromData (const char* name, SBData data, SBType type)
{
    lldb::SBValue sb_value;
    lldb::ValueObjectSP new_value_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        new_value_sp = ValueObjectConstResult::Create (value_sp->GetExecutionContextScope(), 
                                                       type.m_opaque_sp->GetASTContext() ,
                                                       type.m_opaque_sp->GetOpaqueQualType(),
                                                       ConstString(name),
                                                       *data.m_opaque_sp,
                                                       LLDB_INVALID_ADDRESS);
        new_value_sp->SetAddressTypeOfChildren(eAddressTypeLoad);
        sb_value.SetSP(new_value_sp);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (new_value_sp)
            log->Printf ("SBValue(%p)::GetChildFromExpression => \"%s\"", value_sp.get(), new_value_sp->GetName().AsCString());
        else
            log->Printf ("SBValue(%p)::GetChildFromExpression => NULL", value_sp.get());
    }
    return sb_value;
}

SBValue
SBValue::GetChildAtIndex (uint32_t idx)
{
    const bool can_create_synthetic = false;
    lldb::DynamicValueType use_dynamic = eNoDynamicValues;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        use_dynamic = value_sp->GetUpdatePoint().GetTargetSP()->GetPreferDynamicValue();
    return GetChildAtIndex (idx, use_dynamic, can_create_synthetic);
}

SBValue
SBValue::GetChildAtIndex (uint32_t idx, lldb::DynamicValueType use_dynamic, bool can_create_synthetic)
{
    lldb::ValueObjectSP child_sp;

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            const bool can_create = true;
            child_sp = value_sp->GetChildAtIndex (idx, can_create);
            if (can_create_synthetic && !child_sp)
            {
                if (value_sp->IsPointerType())
                {
                    child_sp = value_sp->GetSyntheticArrayMemberFromPointer(idx, can_create);
                }
                else if (value_sp->IsArrayType())
                {
                    child_sp = value_sp->GetSyntheticArrayMemberFromArray(idx, can_create);
                }
            }
                
            if (child_sp)
            {
                if (use_dynamic != lldb::eNoDynamicValues)
                {
                    lldb::ValueObjectSP dynamic_sp(child_sp->GetDynamicValue (use_dynamic));
                    if (dynamic_sp)
                        child_sp = dynamic_sp;
                }
            }
        }
    }
    
    SBValue sb_value (child_sp);
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetChildAtIndex (%u) => SBValue(%p)", value_sp.get(), idx, value_sp.get());

    return sb_value;
}

uint32_t
SBValue::GetIndexOfChildWithName (const char *name)
{
    uint32_t idx = UINT32_MAX;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
        
            idx = value_sp->GetIndexOfChildWithName (ConstString(name));
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (idx == UINT32_MAX)
            log->Printf ("SBValue(%p)::GetIndexOfChildWithName (name=\"%s\") => NOT FOUND", value_sp.get(), name);
        else
            log->Printf ("SBValue(%p)::GetIndexOfChildWithName (name=\"%s\") => %u", value_sp.get(), name, idx);
    }
    return idx;
}

SBValue
SBValue::GetChildMemberWithName (const char *name)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        lldb::DynamicValueType use_dynamic_value = eNoDynamicValues;
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
            use_dynamic_value = target_sp->GetPreferDynamicValue();
        }
        return GetChildMemberWithName (name, use_dynamic_value);
    }
    return SBValue();
}

SBValue
SBValue::GetChildMemberWithName (const char *name, lldb::DynamicValueType use_dynamic_value)
{
    lldb::ValueObjectSP child_sp;
    const ConstString str_name (name);


    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
            child_sp = value_sp->GetChildMemberWithName (str_name, true);
            if (use_dynamic_value != lldb::eNoDynamicValues)
            {
                if (child_sp)
                {
                    lldb::ValueObjectSP dynamic_sp = child_sp->GetDynamicValue (use_dynamic_value);
                    if (dynamic_sp)
                        child_sp = dynamic_sp;
                }
            }
        }
    }
    
    SBValue sb_value (child_sp);

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetChildMemberWithName (name=\"%s\") => SBValue(%p)", value_sp.get(), name, value_sp.get());

    return sb_value;
}

lldb::SBValue
SBValue::GetDynamicValue (lldb::DynamicValueType use_dynamic)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            return SBValue (value_sp->GetDynamicValue(use_dynamic));
        }
    }
    
    return SBValue();
}

lldb::SBValue
SBValue::GetStaticValue ()
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            return SBValue(value_sp->GetStaticValue());
        }
    }
    
    return SBValue();
}

bool
SBValue::IsDynamic()
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            return value_sp->IsDynamic();
        }
    }
    return false;
}

lldb::SBValue
SBValue::GetValueForExpressionPath(const char* expr_path)
{
    lldb::ValueObjectSP child_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            // using default values for all the fancy options, just do it if you can
            child_sp = value_sp->GetValueForExpressionPath(expr_path);
        }
    }
    
    SBValue sb_value (child_sp);
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetValueForExpressionPath (expr_path=\"%s\") => SBValue(%p)", value_sp.get(), expr_path, value_sp.get());
    
    return sb_value;
}

int64_t
SBValue::GetValueAsSigned(SBError& error, int64_t fail_value)
{
    error.Clear();
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            Scalar scalar;
            if (value_sp->ResolveValue (scalar))
                return scalar.GetRawBits64(fail_value);
            else
                error.SetErrorString("could not get value");
        }
        else
            error.SetErrorString("could not get target");
    }
    error.SetErrorString("invalid SBValue");
    return fail_value;
}

uint64_t
SBValue::GetValueAsUnsigned(SBError& error, uint64_t fail_value)
{
    error.Clear();
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            Scalar scalar;
            if (value_sp->ResolveValue (scalar))
                return scalar.GetRawBits64(fail_value);
            else
                error.SetErrorString("could not get value");
        }
        else
            error.SetErrorString("could not get target");
    }
    error.SetErrorString("invalid SBValue");
    return fail_value;
}

int64_t
SBValue::GetValueAsSigned(int64_t fail_value)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            Scalar scalar;
            if (value_sp->ResolveValue (scalar))
                return scalar.GetRawBits64(fail_value);
        }
    }
    return fail_value;
}

uint64_t
SBValue::GetValueAsUnsigned(uint64_t fail_value)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());
            Scalar scalar;
            if (value_sp->ResolveValue (scalar))
                return scalar.GetRawBits64(fail_value);
        }
    }
    return fail_value;
}

uint32_t
SBValue::GetNumChildren ()
{
    uint32_t num_children = 0;

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());

            num_children = value_sp->GetNumChildren();
        }
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetNumChildren () => %u", value_sp.get(), num_children);

    return num_children;
}


SBValue
SBValue::Dereference ()
{
    SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());

            Error error;
            sb_value = value_sp->Dereference (error);
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::Dereference () => SBValue(%p)", value_sp.get(), value_sp.get());

    return sb_value;
}

bool
SBValue::TypeIsPointerType ()
{
    bool is_ptr_type = false;

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());

            is_ptr_type = value_sp->IsPointerType();
        }
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::TypeIsPointerType () => %i", value_sp.get(), is_ptr_type);


    return is_ptr_type;
}

void *
SBValue::GetOpaqueType()
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp(value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (value_sp->GetUpdatePoint().GetTargetSP()->GetAPIMutex());

            return value_sp->GetClangType();
        }
    }
    return NULL;
}

lldb::SBTarget
SBValue::GetTarget()
{
    SBTarget sb_target;
    TargetSP target_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        target_sp = value_sp->GetUpdatePoint().GetTargetSP();
        sb_target.SetSP (target_sp);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (target_sp.get() == NULL)
            log->Printf ("SBValue(%p)::GetTarget () => NULL", value_sp.get());
        else
            log->Printf ("SBValue(%p)::GetTarget () => %p", value_sp.get(), target_sp.get());
    }
    return sb_target;
}

lldb::SBProcess
SBValue::GetProcess()
{
    SBProcess sb_process;
    ProcessSP process_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        process_sp = value_sp->GetUpdatePoint().GetProcessSP();
        if (process_sp)
            sb_process.SetSP (process_sp);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (process_sp.get() == NULL)
            log->Printf ("SBValue(%p)::GetProcess () => NULL", value_sp.get());
        else
            log->Printf ("SBValue(%p)::GetProcess () => %p", value_sp.get(), process_sp.get());
    }
    return sb_process;
}

lldb::SBThread
SBValue::GetThread()
{
    SBThread sb_thread;
    ThreadSP thread_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        if (value_sp->GetExecutionContextScope())
        {
            thread_sp = value_sp->GetExecutionContextScope()->CalculateThread()->shared_from_this();
            sb_thread.SetThread(thread_sp);
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (thread_sp.get() == NULL)
            log->Printf ("SBValue(%p)::GetThread () => NULL", value_sp.get());
        else
            log->Printf ("SBValue(%p)::GetThread () => %p", value_sp.get(), thread_sp.get());
    }
    return sb_thread;
}

lldb::SBFrame
SBValue::GetFrame()
{
    SBFrame sb_frame;
    StackFrameSP frame_sp;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        if (value_sp->GetExecutionContextScope())
        {
            frame_sp = value_sp->GetExecutionContextScope()->CalculateStackFrame()->shared_from_this();
            sb_frame.SetFrameSP (frame_sp);
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (frame_sp.get() == NULL)
            log->Printf ("SBValue(%p)::GetFrame () => NULL", value_sp.get());
        else
            log->Printf ("SBValue(%p)::GetFrame () => %p", value_sp.get(), frame_sp.get());
    }
    return sb_frame;
}


lldb::ValueObjectSP
SBValue::GetSP () const
{
    return m_opaque_sp;
}

void
SBValue::SetSP (const lldb::ValueObjectSP &sp)
{
    m_opaque_sp = sp;
}


bool
SBValue::GetExpressionPath (SBStream &description)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        value_sp->GetExpressionPath (description.ref(), false);
        return true;
    }
    return false;
}

bool
SBValue::GetExpressionPath (SBStream &description, bool qualify_cxx_base_classes)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        value_sp->GetExpressionPath (description.ref(), qualify_cxx_base_classes);
        return true;
    }
    return false;
}

bool
SBValue::GetDescription (SBStream &description)
{
    Stream &strm = description.ref();

    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        ValueObject::DumpValueObject (strm, value_sp.get());
    }
    else
        strm.PutCString ("No value");

    return true;
}

lldb::Format
SBValue::GetFormat ()
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        return value_sp->GetFormat();
    return eFormatDefault;
}

void
SBValue::SetFormat (lldb::Format format)
{
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
        value_sp->SetFormat(format);
}

lldb::SBValue
SBValue::AddressOf()
{
    SBValue sb_value;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        Target* target = value_sp->GetUpdatePoint().GetTargetSP().get();
        if (target)
        {
            Mutex::Locker api_locker (target->GetAPIMutex());
            Error error;
            sb_value = value_sp->AddressOf (error);
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetPointerToObject () => SBValue(%p)", value_sp.get(), value_sp.get());
    
    return sb_value;
}

lldb::addr_t
SBValue::GetLoadAddress()
{
    lldb::addr_t value = LLDB_INVALID_ADDRESS;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        Target* target = value_sp->GetUpdatePoint().GetTargetSP().get();
        if (target)
        {
            Mutex::Locker api_locker (target->GetAPIMutex());
            const bool scalar_is_load_address = true;
            AddressType addr_type;
            value = value_sp->GetAddressOf(scalar_is_load_address, &addr_type);
            if (addr_type == eAddressTypeFile)
            {
                Module* module = value_sp->GetModule();
                if (!module)
                    value = LLDB_INVALID_ADDRESS;
                else
                {
                    Address addr;
                    module->ResolveFileAddress(value, addr);
                    value = addr.GetLoadAddress(value_sp->GetUpdatePoint().GetTargetSP().get());
                }
            }
            else if (addr_type == eAddressTypeHost || addr_type == eAddressTypeInvalid)
                value = LLDB_INVALID_ADDRESS;
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetLoadAddress () => (%llu)", value_sp.get(), value);
    
    return value;
}

lldb::SBAddress
SBValue::GetAddress()
{
    Address addr;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        Target* target = value_sp->GetUpdatePoint().GetTargetSP().get();
        if (target)
        {
            lldb::addr_t value = LLDB_INVALID_ADDRESS;
            Mutex::Locker api_locker (target->GetAPIMutex());
            const bool scalar_is_load_address = true;
            AddressType addr_type;
            value = value_sp->GetAddressOf(scalar_is_load_address, &addr_type);
            if (addr_type == eAddressTypeFile)
            {
                Module* module = value_sp->GetModule();
                if (module)
                    module->ResolveFileAddress(value, addr);
            }
            else if (addr_type == eAddressTypeLoad)
            {
                // no need to check the return value on this.. if it can actually do the resolve
                // addr will be in the form (section,offset), otherwise it will simply be returned
                // as (NULL, value)
                addr.SetLoadAddress(value, target);
            }
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetAddress () => (%s,%llu)", value_sp.get(), (addr.GetSection() ? addr.GetSection()->GetName().GetCString() : "NULL"), addr.GetOffset());
    return SBAddress(new Address(addr));
}

lldb::SBData
SBValue::GetPointeeData (uint32_t item_idx,
                         uint32_t item_count)
{
    lldb::SBData sb_data;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        Target* target = value_sp->GetUpdatePoint().GetTargetSP().get();
        if (target)
        {
			DataExtractorSP data_sp(new DataExtractor());
            Mutex::Locker api_locker (target->GetAPIMutex());
            value_sp->GetPointeeData(*data_sp, item_idx, item_count);
            if (data_sp->GetByteSize() > 0)
                *sb_data = data_sp;
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetPointeeData (%d, %d) => SBData(%p)",
                     value_sp.get(),
                     item_idx,
                     item_count,
                     sb_data.get());
    
    return sb_data;
}

lldb::SBData
SBValue::GetData ()
{
    lldb::SBData sb_data;
    lldb::ValueObjectSP value_sp(GetSP());
    if (value_sp)
    {
        TargetSP target_sp (value_sp->GetUpdatePoint().GetTargetSP());
        if (target_sp)
        {
            Mutex::Locker api_locker (target_sp->GetAPIMutex());
			DataExtractorSP data_sp(new DataExtractor());
            value_sp->GetData(*data_sp);
            if (data_sp->GetByteSize() > 0)
                *sb_data = data_sp;
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetData () => SBData(%p)",
                     value_sp.get(),
                     sb_data.get());
    
    return sb_data;
}

lldb::SBWatchpoint
SBValue::Watch (bool resolve_location, bool read, bool write)
{
    SBWatchpoint sb_watchpoint;
    
    // If the SBValue is not valid, there's no point in even trying to watch it.
    lldb::ValueObjectSP value_sp(GetSP());
    TargetSP target_sp (GetTarget().GetSP());
    if (value_sp && target_sp)
    {
        // Read and Write cannot both be false.
        if (!read && !write)
            return sb_watchpoint;
        
        // If the value is not in scope, don't try and watch and invalid value
        if (!IsInScope())
            return sb_watchpoint;
        
        addr_t addr = GetLoadAddress();
        if (addr == LLDB_INVALID_ADDRESS)
            return sb_watchpoint;
        size_t byte_size = GetByteSize();
        if (byte_size == 0)
            return sb_watchpoint;
                
        uint32_t watch_type = 0;
        if (read)
            watch_type |= LLDB_WATCH_TYPE_READ;
        if (write)
            watch_type |= LLDB_WATCH_TYPE_WRITE;
        
        WatchpointSP watchpoint_sp = target_sp->CreateWatchpoint(addr, byte_size, watch_type);
                
        if (watchpoint_sp) 
        {
            sb_watchpoint.SetSP (watchpoint_sp);
            Declaration decl;
            if (value_sp->GetDeclaration (decl))
            {
                if (decl.GetFile()) 
                {
                    StreamString ss;
                    // True to show fullpath for declaration file.
                    decl.DumpStopContext(&ss, true);
                    watchpoint_sp->SetDeclInfo(ss.GetString());
                }
            }
        }
    }
    return sb_watchpoint;
}

lldb::SBWatchpoint
SBValue::WatchPointee (bool resolve_location, bool read, bool write)
{
    SBWatchpoint sb_watchpoint;
    if (IsInScope() && GetType().IsPointerType())
        sb_watchpoint = Dereference().Watch (resolve_location, read, write);
    return sb_watchpoint;
}

