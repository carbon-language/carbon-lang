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

SBValue::~SBValue()
{
}

bool
SBValue::IsValid () const
{
    return  (m_opaque_sp.get() != NULL);
}

const char *
SBValue::GetName()
{
    if (IsValid())
        return m_opaque_sp->GetName().AsCString();
    else
        return NULL;
}

const char *
SBValue::GetTypeName ()
{
    if (IsValid())
        return m_opaque_sp->GetTypeName().AsCString();
    else
        return NULL;
}

size_t
SBValue::GetByteSize ()
{
    size_t result = 0;

    if (IsValid())
        result = m_opaque_sp->GetByteSize();

    return result;
}

bool
SBValue::IsInScope (const SBFrame &frame)
{
    bool result = false;

    if (IsValid())
        result = m_opaque_sp->IsInScope (frame.get());

    return result;
}

const char *
SBValue::GetValue (const SBFrame &frame)
{
    const char *value_string = NULL;
    if ( m_opaque_sp)
        value_string = m_opaque_sp->GetValueAsCString (frame.get());
    return value_string;
}

const char *
SBValue::GetObjectDescription (const SBFrame &frame)
{
    const char *value_string = NULL;
    if ( m_opaque_sp)
        value_string = m_opaque_sp->GetObjectDescription (frame.get());
    return value_string;
}

bool
SBValue::GetValueDidChange (const SBFrame &frame)
{
    if (IsValid())
        return m_opaque_sp->GetValueDidChange (frame.get());
    return false;
}

const char *
SBValue::GetSummary (const SBFrame &frame)
{
    const char *value_string = NULL;
    if ( m_opaque_sp)
        value_string = m_opaque_sp->GetSummaryAsCString(frame.get());
    return value_string;
}

const char *
SBValue::GetLocation (const SBFrame &frame)
{
    const char *value_string = NULL;
    if (IsValid())
        value_string = m_opaque_sp->GetLocationAsCString(frame.get());
    return value_string;
}

bool
SBValue::SetValueFromCString (const SBFrame &frame, const char *value_str)
{
    bool success = false;
    if (IsValid())
        success = m_opaque_sp->SetValueFromCString (frame.get(), value_str);
    return success;
}

SBValue
SBValue::GetChildAtIndex (uint32_t idx)
{
    lldb::ValueObjectSP child_sp;

    if (IsValid())
    {
        child_sp = m_opaque_sp->GetChildAtIndex (idx, true);
    }

    SBValue sb_value (child_sp);
    return sb_value;
}

uint32_t
SBValue::GetIndexOfChildWithName (const char *name)
{
    if (IsValid())
        return m_opaque_sp->GetIndexOfChildWithName (ConstString(name));
    return UINT32_MAX;
}

SBValue
SBValue::GetChildMemberWithName (const char *name)
{
    lldb::ValueObjectSP child_sp;
    const ConstString str_name (name);

    if (IsValid())
    {
        child_sp = m_opaque_sp->GetChildMemberWithName (str_name, true);
    }

    SBValue sb_value (child_sp);
    return sb_value;
}


uint32_t
SBValue::GetNumChildren ()
{
    uint32_t num_children = 0;

    if (IsValid())
    {
        num_children = m_opaque_sp->GetNumChildren();
    }

    return num_children;
}

bool
SBValue::ValueIsStale ()
{
    bool result = true;

    if (IsValid())
    {
        result = m_opaque_sp->GetValueIsValid();
    }

    return result;
}


SBValue
SBValue::Dereference ()
{
    if (IsValid())
    {
        if (m_opaque_sp->IsPointerType())
        {
            return GetChildAtIndex(0);
        }
    }
    return *this;
}

bool
SBValue::TypeIsPtrType ()
{
    bool is_ptr_type = false;

    if (IsValid())
    {
        is_ptr_type = m_opaque_sp->IsPointerType();
    }

    return is_ptr_type;
}

void *
SBValue::GetOpaqueType()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetOpaqueClangQualType();
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
SBValue::GetDescription (SBStream &description)
{
    if (m_opaque_sp)
    {
        const char *name = GetName();
        const char *type_name = GetTypeName ();
        size_t byte_size = GetByteSize ();
        uint32_t num_children = GetNumChildren ();
        bool is_stale = ValueIsStale ();
        description.Printf ("name: '%s', type: %s, size: %d", (name != NULL ? name : "<unknown name>"),
                            (type_name != NULL ? type_name : "<unknown type name>"), (int) byte_size);
        if (num_children > 0)
            description.Printf (", num_children: %d", num_children);

        if (is_stale)
            description.Printf (" [value is stale]");
    }
    else
        description.Printf ("No value");

    return true;
}
