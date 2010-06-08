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
#include "lldb/Symbol/Function.h"

using namespace lldb;


SBFunction::SBFunction () :
    m_lldb_object_ptr (NULL)
{
}

SBFunction::SBFunction (lldb_private::Function *lldb_object_ptr) :
    m_lldb_object_ptr (lldb_object_ptr)
{
}

SBFunction::~SBFunction ()
{
    m_lldb_object_ptr = NULL;
}

bool
SBFunction::IsValid () const
{
    return m_lldb_object_ptr != NULL;
}

const char *
SBFunction::GetName() const
{
    if (m_lldb_object_ptr)
        return m_lldb_object_ptr->GetMangled().GetName().AsCString();
    return NULL;
}

const char *
SBFunction::GetMangledName () const
{
    if (m_lldb_object_ptr)
        return m_lldb_object_ptr->GetMangled().GetMangledName().AsCString();
    return NULL;
}

bool
SBFunction::operator == (const SBFunction &rhs) const
{
    return m_lldb_object_ptr == rhs.m_lldb_object_ptr;
}

bool
SBFunction::operator != (const SBFunction &rhs) const
{
    return m_lldb_object_ptr != rhs.m_lldb_object_ptr;
}
