//===-- SBSymbol.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSymbol.h"
#include "lldb/API/SBStream.h"
#include "lldb/Symbol/Symbol.h"

using namespace lldb;


SBSymbol::SBSymbol () :
    m_opaque_ptr (NULL)
{
}

SBSymbol::SBSymbol (lldb_private::Symbol *lldb_object_ptr) :
    m_opaque_ptr (lldb_object_ptr)
{
}

SBSymbol::~SBSymbol ()
{
    m_opaque_ptr = NULL;
}

bool
SBSymbol::IsValid () const
{
    return m_opaque_ptr != NULL;
}

const char *
SBSymbol::GetName() const
{
    if (m_opaque_ptr)
        return m_opaque_ptr->GetMangled().GetName().AsCString();
    return NULL;
}

const char *
SBSymbol::GetMangledName () const
{
    if (m_opaque_ptr)
        return m_opaque_ptr->GetMangled().GetMangledName().AsCString();
    return NULL;
}


bool
SBSymbol::operator == (const SBSymbol &rhs) const
{
    return m_opaque_ptr == rhs.m_opaque_ptr;
}

bool
SBSymbol::operator != (const SBSymbol &rhs) const
{
    return m_opaque_ptr != rhs.m_opaque_ptr;
}

bool
SBSymbol::GetDescription (SBStream &description)
{
    if (m_opaque_ptr)
    {
        description.ref();
        m_opaque_ptr->GetDescription (description.get(), 
                                      lldb::eDescriptionLevelFull, NULL);
    }
    else
        description.Printf ("No value");
    
    return true;
}
