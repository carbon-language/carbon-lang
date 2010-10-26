//===-- SBError.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBError.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"

#include <stdarg.h>

using namespace lldb;
using namespace lldb_private;


SBError::SBError () :
    m_opaque_ap ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBError::SBError () ==> this = %p", this);
}

SBError::SBError (const SBError &rhs) :
    m_opaque_ap ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (rhs.IsValid())
        m_opaque_ap.reset (new Error(*rhs));

    if (log)
    {
        SBStream sstr;
        GetDescription (sstr);
        log->Printf ("SBError::SBError (const SBError &rhs) rhs.m_opaque_ap.get() = %p ==> this = %p (%s)",
                     (rhs.IsValid() ? rhs.m_opaque_ap.get() : NULL), this, sstr.GetData());
    }
}


SBError::~SBError()
{
}

const SBError &
SBError::operator = (const SBError &rhs)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
    {
        SBStream sstr;
        rhs.GetDescription (sstr);
        log->Printf ("SBError::operator= (const SBError &rhs) rhs.m_opaque_ap.get() = %p (%s)", 
                     (rhs.IsValid() ? rhs.m_opaque_ap.get() : NULL), sstr.GetData());
    }

    if (rhs.IsValid())
    {
        if (m_opaque_ap.get())
            *m_opaque_ap = *rhs;
        else
            m_opaque_ap.reset (new Error(*rhs));
    }
    else
    {
        m_opaque_ap.reset();
    }

    if (log)
        log->Printf ("SBError::operator= ==> this = %p", this);

    return *this;
}


const char *
SBError::GetCString () const
{
    if (m_opaque_ap.get())
        return m_opaque_ap->AsCString();
    return NULL;
}

void
SBError::Clear ()
{
    if (m_opaque_ap.get())
        m_opaque_ap->Clear();
}

bool
SBError::Fail () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBError::Fail ()");

    bool ret_value = false;
    if (m_opaque_ap.get())
        ret_value = m_opaque_ap->Fail();

    if (log)
        log->Printf ("SBError::Fail ==> %s", (ret_value ? "true" : "false"));

    return ret_value;
}

bool
SBError::Success () const
{
    if (m_opaque_ap.get())
        return m_opaque_ap->Success();
    return false;
}

uint32_t
SBError::GetError () const
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetError();
    return true;
}

ErrorType
SBError::GetType () const
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetType();
    return eErrorTypeInvalid;
}

void
SBError::SetError (uint32_t err, ErrorType type)
{
    CreateIfNeeded ();
    m_opaque_ap->SetError (err, type);
}

void
SBError::SetError (const Error &lldb_error)
{
    CreateIfNeeded ();
    *m_opaque_ap = lldb_error;
}


void
SBError::SetErrorToErrno ()
{
    CreateIfNeeded ();
    m_opaque_ap->SetErrorToErrno ();
}

void
SBError::SetErrorToGenericError ()
{
    CreateIfNeeded ();
    m_opaque_ap->SetErrorToErrno ();
}

void
SBError::SetErrorString (const char *err_str)
{
    CreateIfNeeded ();
    m_opaque_ap->SetErrorString (err_str);
}

int
SBError::SetErrorStringWithFormat (const char *format, ...)
{
    CreateIfNeeded ();
    va_list args;
    va_start (args, format);
    int num_chars = m_opaque_ap->SetErrorStringWithVarArg (format, args);
    va_end (args);
    return num_chars;
}

bool
SBError::IsValid () const
{
    return m_opaque_ap.get() != NULL;
}

void
SBError::CreateIfNeeded ()
{
    if (m_opaque_ap.get() == NULL)
        m_opaque_ap.reset(new Error ());
}


lldb_private::Error *
SBError::operator->()
{
    return m_opaque_ap.get();
}

lldb_private::Error *
SBError::get()
{
    return m_opaque_ap.get();
}


const lldb_private::Error &
SBError::operator*() const
{
    // Be sure to call "IsValid()" before calling this function or it will crash
    return *m_opaque_ap;
}

bool
SBError::GetDescription (SBStream &description)
{
    if (m_opaque_ap.get())
    {
        if (Success())
            description.Printf ("Status: Success");
        else
        {
            const char * err_string = GetCString();
            description.Printf ("Status:  Error: %s",  (err_string != NULL ? err_string : ""));
        }
    }
    else
        description.Printf ("No value");

    return true;
} 

bool
SBError::GetDescription (SBStream &description) const
{
    if (m_opaque_ap.get())
    {
        if (Success())
            description.Printf ("Status: Success");
        else
        {
            const char * err_string = GetCString();
            description.Printf ("Status:  Error: %s",  (err_string != NULL ? err_string : ""));
        }
    }
    else
        description.Printf ("No value");

    return true;
} 
