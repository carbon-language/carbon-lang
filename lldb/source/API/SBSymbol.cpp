//===-- SBSymbol.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSymbol.h"
#include "lldb/Symbol/Symbol.h"

using namespace lldb;


SBSymbol::SBSymbol () :
    m_lldb_object_ptr (NULL)
{
}

SBSymbol::SBSymbol (lldb_private::Symbol *lldb_object_ptr) :
    m_lldb_object_ptr (lldb_object_ptr)
{
}

SBSymbol::~SBSymbol ()
{
    m_lldb_object_ptr = NULL;
}

bool
SBSymbol::IsValid () const
{
    return m_lldb_object_ptr != NULL;
}

const char *
SBSymbol::GetName() const
{
    if (m_lldb_object_ptr)
        return m_lldb_object_ptr->GetMangled().GetName().AsCString();
    return NULL;
}

const char *
SBSymbol::GetMangledName () const
{
    if (m_lldb_object_ptr)
        return m_lldb_object_ptr->GetMangled().GetMangledName().AsCString();
    return NULL;
}


bool
SBSymbol::operator == (const SBSymbol &rhs) const
{
    return m_lldb_object_ptr == rhs.m_lldb_object_ptr;
}

bool
SBSymbol::operator != (const SBSymbol &rhs) const
{
    return m_lldb_object_ptr != rhs.m_lldb_object_ptr;
}
