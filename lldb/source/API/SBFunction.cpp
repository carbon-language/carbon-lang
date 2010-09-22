//===-- SBFunction.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBFunction.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/Symbol/Function.h"

using namespace lldb;


SBFunction::SBFunction () :
    m_opaque_ptr (NULL)
{
}

SBFunction::SBFunction (lldb_private::Function *lldb_object_ptr) :
    m_opaque_ptr (lldb_object_ptr)
{
}

SBFunction::~SBFunction ()
{
    m_opaque_ptr = NULL;
}

bool
SBFunction::IsValid () const
{
    return m_opaque_ptr != NULL;
}

const char *
SBFunction::GetName() const
{
    if (m_opaque_ptr)
        return m_opaque_ptr->GetMangled().GetName().AsCString();
    return NULL;
}

const char *
SBFunction::GetMangledName () const
{
    if (m_opaque_ptr)
        return m_opaque_ptr->GetMangled().GetMangledName().AsCString();
    return NULL;
}

bool
SBFunction::operator == (const SBFunction &rhs) const
{
    return m_opaque_ptr == rhs.m_opaque_ptr;
}

bool
SBFunction::operator != (const SBFunction &rhs) const
{
    return m_opaque_ptr != rhs.m_opaque_ptr;
}

bool
SBFunction::GetDescription (SBStream &description)
{
    if (m_opaque_ptr)
    {
        description.ref();
        m_opaque_ptr->Dump (description.get(), false);
    }
    else
        description.Printf ("No value");

    return true;
}
