//===-- SBExpressionOptions.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBExpressionOptions.h"
#include "lldb/API/SBStream.h"

#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;


SBExpressionOptions::SBExpressionOptions ()
{
    m_opaque_ap.reset(new EvaluateExpressionOptions());
}

SBExpressionOptions::SBExpressionOptions (bool coerce_to_id,
                         bool unwind_on_error,
                         bool keep_in_memory,
                         bool run_others,
                         DynamicValueType use_dynamic,
                         uint32_t timeout_usec)
{
    m_opaque_ap.reset(new EvaluateExpressionOptions());
    m_opaque_ap->SetCoerceToId(coerce_to_id);
    m_opaque_ap->SetUnwindOnError(unwind_on_error);
    m_opaque_ap->SetKeepInMemory(keep_in_memory);
    m_opaque_ap->SetRunOthers(run_others);
    m_opaque_ap->SetUseDynamic (use_dynamic);
    m_opaque_ap->SetTimeoutUsec (timeout_usec);
}

SBExpressionOptions::SBExpressionOptions (const SBExpressionOptions &rhs)
{
    m_opaque_ap.reset(new EvaluateExpressionOptions());
    *(m_opaque_ap.get()) = rhs.ref();
}

const SBExpressionOptions &
SBExpressionOptions::operator = (const SBExpressionOptions &rhs)
{
    if (this != &rhs)
    {
        this->ref() = rhs.ref();
    }
    return *this;
}

SBExpressionOptions::~SBExpressionOptions()
{
}

bool
SBExpressionOptions::DoesCoerceToId () const
{
    return m_opaque_ap->DoesCoerceToId ();
}

void
SBExpressionOptions::SetCoerceToId (bool coerce)
{
    m_opaque_ap->SetCoerceToId (coerce);
}

bool
SBExpressionOptions::DoesUnwindOnError () const
{
    return m_opaque_ap->DoesUnwindOnError ();
}

void
SBExpressionOptions::SetUnwindOnError (bool unwind)
{
    m_opaque_ap->SetUnwindOnError (unwind);
}

bool
SBExpressionOptions::DoesKeepInMemory () const
{
    return m_opaque_ap->DoesKeepInMemory ();
}

void
SBExpressionOptions::SetKeepInMemory (bool keep)
{
    m_opaque_ap->SetKeepInMemory (keep);
}

lldb::DynamicValueType
SBExpressionOptions::GetUseDynamic () const
{
    return m_opaque_ap->GetUseDynamic ();
}

void
SBExpressionOptions::SetUseDynamic (lldb::DynamicValueType dynamic)
{
    m_opaque_ap->SetUseDynamic (dynamic);
}

uint32_t
SBExpressionOptions::GetTimeoutUsec () const
{
    return m_opaque_ap->GetTimeoutUsec ();
}

void
SBExpressionOptions::SetTimeoutUsec (uint32_t timeout)
{
    m_opaque_ap->SetTimeoutUsec (timeout);
}

bool
SBExpressionOptions::GetRunOthers () const
{
    return m_opaque_ap->GetRunOthers ();
}

void
SBExpressionOptions::SetRunOthers (bool run_others)
{
    m_opaque_ap->SetRunOthers (run_others);
}

EvaluateExpressionOptions *
SBExpressionOptions::get() const
{
    return m_opaque_ap.get();
}

EvaluateExpressionOptions &
SBExpressionOptions::ref () const
{
    return *(m_opaque_ap.get());
}
