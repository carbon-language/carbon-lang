//===-- SBCommandReturnObject.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandReturnObject.h"

#include "lldb/API/SBCommandReturnObject.h"

using namespace lldb;

SBCommandReturnObject::SBCommandReturnObject () :
    m_return_object_ap (new lldb_private::CommandReturnObject ())
{
}

SBCommandReturnObject::~SBCommandReturnObject ()
{
    // m_return_object_ap will automatically delete any pointer it owns
}

bool
SBCommandReturnObject::IsValid() const
{
    return m_return_object_ap.get() != NULL;
}


const char *
SBCommandReturnObject::GetOutput ()
{
    if (m_return_object_ap.get())
        return m_return_object_ap->GetOutputStream().GetData();
    return NULL;
}

const char *
SBCommandReturnObject::GetError ()
{
    if (m_return_object_ap.get())
        return m_return_object_ap->GetErrorStream().GetData();
    return NULL;
}

size_t
SBCommandReturnObject::GetOutputSize ()
{
    if (m_return_object_ap.get())
        return m_return_object_ap->GetOutputStream().GetSize();
    return 0;
}

size_t
SBCommandReturnObject::GetErrorSize ()
{
    if (m_return_object_ap.get())
        return m_return_object_ap->GetErrorStream().GetSize();
    return 0;
}

size_t
SBCommandReturnObject::PutOutput (FILE *fh)
{
    if (fh)
    {
        size_t num_bytes = GetOutputSize ();
        if (num_bytes)
            return ::fprintf (fh, "%s", GetOutput());
    }
    return 0;
}

size_t
SBCommandReturnObject::PutError (FILE *fh)
{
    if (fh)
    {
        size_t num_bytes = GetErrorSize ();
        if (num_bytes)
            return ::fprintf (fh, "%s", GetError());
    }
    return 0;
}

void
SBCommandReturnObject::Clear()
{
    if (m_return_object_ap.get())
        m_return_object_ap->Clear();
}

lldb::ReturnStatus
SBCommandReturnObject::GetStatus()
{
    if (m_return_object_ap.get())
        return m_return_object_ap->GetStatus();
    return lldb::eReturnStatusInvalid;
}

bool
SBCommandReturnObject::Succeeded ()
{
    if (m_return_object_ap.get())
        return m_return_object_ap->Succeeded();
    return false;
}

bool
SBCommandReturnObject::HasResult ()
{
    if (m_return_object_ap.get())
        return m_return_object_ap->HasResult();
    return false;
}

void
SBCommandReturnObject::AppendMessage (const char *message)
{
    if (m_return_object_ap.get())
        m_return_object_ap->AppendMessage (message);
}

lldb_private::CommandReturnObject *
SBCommandReturnObject::GetLLDBObjectPtr()
{
    return m_return_object_ap.get();
}


lldb_private::CommandReturnObject &
SBCommandReturnObject::GetLLDBObjectRef()
{
    assert(m_return_object_ap.get());
    return *(m_return_object_ap.get());
}


void
SBCommandReturnObject::SetLLDBObjectPtr (lldb_private::CommandReturnObject *ptr)
{
    if (m_return_object_ap.get())
        m_return_object_ap.reset (ptr);
}

