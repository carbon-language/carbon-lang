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

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Variable.h"
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

const SBValue &
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
SBValue::IsValid () const
{
    // If this function ever changes to anything that does more than just
    // check if the opaque shared pointer is non NULL, then we need to update
    // all "if (m_opaque_sp)" code in this file.
    return m_opaque_sp.get() != NULL;
}

SBError
SBValue::GetError()
{
    SBError sb_error;
    
    if (m_opaque_sp.get())
        sb_error.SetError(m_opaque_sp->GetError());
    
    return sb_error;
}

const char *
SBValue::GetName()
{

    const char *name = NULL;
    if (m_opaque_sp)
        name = m_opaque_sp->GetName().GetCString();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (name)
            log->Printf ("SBValue(%p)::GetName () => \"%s\"", m_opaque_sp.get(), name);
        else
            log->Printf ("SBValue(%p)::GetName () => NULL", m_opaque_sp.get(), name);
    }

    return name;
}

const char *
SBValue::GetTypeName ()
{
    const char *name = NULL;
    if (m_opaque_sp)
        name = m_opaque_sp->GetTypeName().GetCString();
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (name)
            log->Printf ("SBValue(%p)::GetTypeName () => \"%s\"", m_opaque_sp.get(), name);
        else
            log->Printf ("SBValue(%p)::GetTypeName () => NULL", m_opaque_sp.get());
    }

    return name;
}

size_t
SBValue::GetByteSize ()
{
    size_t result = 0;

    if (m_opaque_sp)
        result = m_opaque_sp->GetByteSize();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetByteSize () => %zu", m_opaque_sp.get(), result);

    return result;
}

bool
SBValue::IsInScope (const SBFrame &sb_frame)
{
    return IsInScope();
}

bool
SBValue::IsInScope ()
{
    bool result = false;

    if (m_opaque_sp)
    {
        if (m_opaque_sp->GetUpdatePoint().GetTarget())
            Mutex::Locker api_locker (m_opaque_sp->GetUpdatePoint().GetTarget()->GetAPIMutex());
        result = m_opaque_sp->IsInScope ();
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::IsInScope () => %i", m_opaque_sp.get(), result);

    return result;
}

const char *
SBValue::GetValue (const SBFrame &sb_frame)
{
    return GetValue();
}

const char *
SBValue::GetValue ()
{
    const char *cstr = NULL;
    if (m_opaque_sp)
    {
        if (m_opaque_sp->GetUpdatePoint().GetTarget())
            Mutex::Locker api_locker (m_opaque_sp->GetUpdatePoint().GetTarget()->GetAPIMutex());
        cstr = m_opaque_sp->GetValueAsCString ();
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetValue => \"%s\"", m_opaque_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetValue => NULL", m_opaque_sp.get());
    }

    return cstr;
}

ValueType
SBValue::GetValueType ()
{
    ValueType result = eValueTypeInvalid;
    if (m_opaque_sp)
        result = m_opaque_sp->GetValueType();
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        switch (result)
        {
        case eValueTypeInvalid:         log->Printf ("SBValue(%p)::GetValueType () => eValueTypeInvalid", m_opaque_sp.get()); break;
        case eValueTypeVariableGlobal:  log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableGlobal", m_opaque_sp.get()); break;
        case eValueTypeVariableStatic:  log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableStatic", m_opaque_sp.get()); break;
        case eValueTypeVariableArgument:log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableArgument", m_opaque_sp.get()); break;
        case eValueTypeVariableLocal:   log->Printf ("SBValue(%p)::GetValueType () => eValueTypeVariableLocal", m_opaque_sp.get()); break;
        case eValueTypeRegister:        log->Printf ("SBValue(%p)::GetValueType () => eValueTypeRegister", m_opaque_sp.get()); break;
        case eValueTypeRegisterSet:     log->Printf ("SBValue(%p)::GetValueType () => eValueTypeRegisterSet", m_opaque_sp.get()); break;
        case eValueTypeConstResult:     log->Printf ("SBValue(%p)::GetValueType () => eValueTypeConstResult", m_opaque_sp.get()); break;
        default:     log->Printf ("SBValue(%p)::GetValueType () => %i ???", m_opaque_sp.get(), result); break;
        }
    }
    return result;
}

const char *
SBValue::GetObjectDescription (const SBFrame &sb_frame)
{
    return GetObjectDescription ();
}

const char *
SBValue::GetObjectDescription ()
{
    const char *cstr = NULL;
    if (m_opaque_sp)
    {
        if (m_opaque_sp->GetUpdatePoint().GetTarget())
            Mutex::Locker api_locker (m_opaque_sp->GetUpdatePoint().GetTarget()->GetAPIMutex());
        cstr = m_opaque_sp->GetObjectDescription ();
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetObjectDescription => \"%s\"", m_opaque_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetObjectDescription => NULL", m_opaque_sp.get());
    }
    return cstr;
}

bool
SBValue::GetValueDidChange (const SBFrame &sb_frame)
{
    return GetValueDidChange ();
}

bool
SBValue::GetValueDidChange ()
{
    bool result = false;
    if (m_opaque_sp)
    {
        if (m_opaque_sp->GetUpdatePoint().GetTarget())
            Mutex::Locker api_locker (m_opaque_sp->GetUpdatePoint().GetTarget()->GetAPIMutex());
        result = m_opaque_sp->GetValueDidChange ();
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetValueDidChange => %i", m_opaque_sp.get(), result);

    return result;
}

const char *
SBValue::GetSummary (const SBFrame &sb_frame)
{
    return GetSummary ();
}

const char *
SBValue::GetSummary ()
{
    const char *cstr = NULL;
    if (m_opaque_sp)
    {
        if (m_opaque_sp->GetUpdatePoint().GetTarget())
            Mutex::Locker api_locker (m_opaque_sp->GetUpdatePoint().GetTarget()->GetAPIMutex());
        cstr = m_opaque_sp->GetSummaryAsCString();
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetSummary => \"%s\"", m_opaque_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetSummary => NULL", m_opaque_sp.get());
    }
    return cstr;
}

const char *
SBValue::GetLocation (const SBFrame &sb_frame)
{
    return GetLocation ();
}

const char *
SBValue::GetLocation ()
{
    const char *cstr = NULL;
    if (m_opaque_sp)
    {
        if (m_opaque_sp->GetUpdatePoint().GetTarget())
            Mutex::Locker api_locker (m_opaque_sp->GetUpdatePoint().GetTarget()->GetAPIMutex());
        cstr = m_opaque_sp->GetLocationAsCString();
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (cstr)
            log->Printf ("SBValue(%p)::GetSummary => \"%s\"", m_opaque_sp.get(), cstr);
        else
            log->Printf ("SBValue(%p)::GetSummary => NULL", m_opaque_sp.get());
    }
    return cstr;
}

bool
SBValue::SetValueFromCString (const SBFrame &sb_frame, const char *value_str)
{
    return SetValueFromCString (value_str);
}

bool
SBValue::SetValueFromCString (const char *value_str)
{
    bool success = false;
    if (m_opaque_sp)
    {
        if (m_opaque_sp->GetUpdatePoint().GetTarget())
            Mutex::Locker api_locker (m_opaque_sp->GetUpdatePoint().GetTarget()->GetAPIMutex());
        success = m_opaque_sp->SetValueFromCString (value_str);
    }
    return success;
}

SBValue
SBValue::GetChildAtIndex (uint32_t idx)
{
    bool use_dynamic_value = m_opaque_sp->GetUpdatePoint().GetTarget()->GetPreferDynamicValue();
    return GetChildAtIndex (idx, use_dynamic_value);
}

SBValue
SBValue::GetChildAtIndex (uint32_t idx, bool use_dynamic_value)
{
    lldb::ValueObjectSP child_sp;

    if (m_opaque_sp)
    {
        child_sp = m_opaque_sp->GetChildAtIndex (idx, true);
    }

    if (use_dynamic_value)
    {
        if (child_sp)
        {
            lldb::ValueObjectSP dynamic_sp = child_sp->GetDynamicValue(true, child_sp);
            if (dynamic_sp)
                child_sp = dynamic_sp;
        }
    }
    
    SBValue sb_value (child_sp);
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetChildAtIndex (%u) => SBValue(%p)", m_opaque_sp.get(), idx, sb_value.get());

    return sb_value;
}

uint32_t
SBValue::GetIndexOfChildWithName (const char *name)
{
    uint32_t idx = UINT32_MAX;
    if (m_opaque_sp)
        idx = m_opaque_sp->GetIndexOfChildWithName (ConstString(name));
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (idx == UINT32_MAX)
            log->Printf ("SBValue(%p)::GetIndexOfChildWithName (name=\"%s\") => NOT FOUND", m_opaque_sp.get(), name, idx);
        else
            log->Printf ("SBValue(%p)::GetIndexOfChildWithName (name=\"%s\") => %u", m_opaque_sp.get(), name, idx);
    }
    return idx;
}

SBValue
SBValue::GetChildMemberWithName (const char *name)
{
    bool use_dynamic_value = m_opaque_sp->GetUpdatePoint().GetTarget()->GetPreferDynamicValue();
    return GetChildMemberWithName (name, use_dynamic_value);
}

SBValue
SBValue::GetChildMemberWithName (const char *name, bool use_dynamic_value)
{
    lldb::ValueObjectSP child_sp;
    const ConstString str_name (name);

    if (m_opaque_sp)
    {
        child_sp = m_opaque_sp->GetChildMemberWithName (str_name, true);
    }

    if (use_dynamic_value)
    {
        if (child_sp)
        {
            lldb::ValueObjectSP dynamic_sp = child_sp->GetDynamicValue(true, child_sp);
            if (dynamic_sp)
                child_sp = dynamic_sp;
        }
    }
    
    SBValue sb_value (child_sp);

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetChildMemberWithName (name=\"%s\") => SBValue(%p)", m_opaque_sp.get(), name, sb_value.get());

    return sb_value;
}


uint32_t
SBValue::GetNumChildren ()
{
    uint32_t num_children = 0;

    if (m_opaque_sp)
        num_children = m_opaque_sp->GetNumChildren();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::GetNumChildren () => %u", m_opaque_sp.get(), num_children);

    return num_children;
}


SBValue
SBValue::Dereference ()
{
    SBValue sb_value;
    if (m_opaque_sp)
    {
        Error error;
        sb_value = m_opaque_sp->Dereference (error);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::Dereference () => SBValue(%p)", m_opaque_sp.get(), sb_value.get());

    return sb_value;
}

bool
SBValue::TypeIsPointerType ()
{
    bool is_ptr_type = false;

    if (m_opaque_sp)
        is_ptr_type = m_opaque_sp->IsPointerType();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBValue(%p)::TypeIsPointerType () => %i", m_opaque_sp.get(), is_ptr_type);


    return is_ptr_type;
}

void *
SBValue::GetOpaqueType()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetClangType();
    return NULL;
}

// Mimic shared pointer...
lldb_private::ValueObject *
SBValue::get() const
{
    return m_opaque_sp.get();
}

lldb_private::ValueObject *
SBValue::operator->() const
{
    return m_opaque_sp.get();
}

lldb::ValueObjectSP &
SBValue::operator*()
{
    return m_opaque_sp;
}

const lldb::ValueObjectSP &
SBValue::operator*() const
{
    return m_opaque_sp;
}

bool
SBValue::GetExpressionPath (SBStream &description)
{
    if (m_opaque_sp)
    {
        m_opaque_sp->GetExpressionPath (description.ref(), false);
        return true;
    }
    return false;
}

bool
SBValue::GetExpressionPath (SBStream &description, bool qualify_cxx_base_classes)
{
    if (m_opaque_sp)
    {
        m_opaque_sp->GetExpressionPath (description.ref(), qualify_cxx_base_classes);
        return true;
    }
    return false;
}

bool
SBValue::GetDescription (SBStream &description)
{
    if (m_opaque_sp)
    {
        // Don't call all these APIs and cause more logging!
//        const char *name = GetName();
//        const char *type_name = GetTypeName ();
//        size_t byte_size = GetByteSize ();
//        uint32_t num_children = GetNumChildren ();
//        bool is_stale = ValueIsStale ();
//        description.Printf ("name: '%s', type: %s, size: %d", (name != NULL ? name : "<unknown name>"),
//                            (type_name != NULL ? type_name : "<unknown type name>"), (int) byte_size);
//        if (num_children > 0)
//            description.Printf (", num_children: %d", num_children);
//
//        if (is_stale)
//            description.Printf (" [value is stale]");
        
        description.Printf ("name: '%s'", m_opaque_sp->GetName().GetCString());
    }
    else
        description.Printf ("No value");

    return true;
}

lldb::Format
SBValue::GetFormat () const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetFormat();
    return eFormatDefault;
}

void
SBValue::SetFormat (lldb::Format format)
{
    if (m_opaque_sp)
        m_opaque_sp->SetFormat(format);
}

