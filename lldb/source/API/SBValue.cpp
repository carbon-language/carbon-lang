//===-- SBValue.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBValue.h"

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

void
SBValue::Print (FILE *out_file, SBFrame *frame, bool print_type, bool print_value)
{
    if (out_file == NULL)
        return;

    if (IsValid())
    {

        SBThread sb_thread = frame->GetThread();
        SBProcess sb_process = sb_thread.GetProcess();

        lldb_private::StackFrame *lldb_frame = frame->GetLLDBObjectPtr();
        lldb_private::Thread *lldb_thread = sb_thread.GetLLDBObjectPtr();
        lldb_private::Process *lldb_process = sb_process.get();

        lldb_private::ExecutionContext context (lldb_process, lldb_thread, lldb_frame);

        lldb_private::StreamFile out_stream (out_file);

        out_stream.Printf ("%s ", m_opaque_sp->GetName().AsCString (NULL));
        if (! m_opaque_sp->IsInScope (lldb_frame))
            out_stream.Printf ("[out-of-scope] ");
        if (print_type)
        {
            out_stream.Printf ("(%s) ", m_opaque_sp->GetTypeName().AsCString ("<unknown-type>"));
        }

        if (print_value)
        {
            ExecutionContextScope *exe_scope = frame->get();
            const char *val_cstr = m_opaque_sp->GetValueAsCString(exe_scope);
            const char *err_cstr = m_opaque_sp->GetError().AsCString();

            if (!err_cstr)
            {
                const char *sum_cstr = m_opaque_sp->GetSummaryAsCString(exe_scope);
                const bool is_aggregate =
                ClangASTContext::IsAggregateType (m_opaque_sp->GetOpaqueClangQualType());
                if (val_cstr)
                    out_stream.Printf ("= %s ", val_cstr);

                if (sum_cstr)
                    out_stream.Printf ("%s ", sum_cstr);

                if (is_aggregate)
                {
                    out_stream.PutChar ('{');
                    const uint32_t num_children = m_opaque_sp->GetNumChildren();
                    if (num_children)
                    {
                        out_stream.IndentMore();
                        for (uint32_t idx = 0; idx < num_children; ++idx)
                        {
                            lldb::ValueObjectSP child_sp (m_opaque_sp->GetChildAtIndex (idx, true));
                            if (child_sp.get())
                            {
                                out_stream.EOL();
                                out_stream.Indent();
                                out_stream.Printf ("%s (%s) = %s", child_sp.get()->GetName().AsCString (""),
                                                   child_sp.get()->GetTypeName().AsCString ("<unknown type>"),
                                                   child_sp.get()->GetValueAsCString(exe_scope));
                            }
                        }
                        out_stream.IndentLess();
                    }
                    out_stream.EOL();
                    out_stream.Indent ("}");
                }
            }
        }
        out_stream.EOL ();
    }
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
        value_string = m_opaque_sp->GetValueAsCString(frame.get());
    return value_string;
}

bool
SBValue::GetValueDidChange ()
{
    if (IsValid())
        return m_opaque_sp->GetValueDidChange();
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
