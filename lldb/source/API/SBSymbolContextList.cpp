//===-- SBSymbolContextList.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSymbolContextList.h"
#include "lldb/Symbol/SymbolContext.h"

using namespace lldb;
using namespace lldb_private;

SBSymbolContextList::SBSymbolContextList () :
    m_opaque_ap ()
{
}

SBSymbolContextList::SBSymbolContextList (const SBSymbolContextList& rhs) :
    m_opaque_ap ()
{
    if (rhs.IsValid())
        *m_opaque_ap = *rhs.m_opaque_ap;
}

SBSymbolContextList::~SBSymbolContextList ()
{
}

const SBSymbolContextList &
SBSymbolContextList::operator = (const SBSymbolContextList &rhs)
{
    if (this != &rhs)
    {
        if (rhs.IsValid())
            m_opaque_ap.reset (new lldb_private::SymbolContextList(*rhs.m_opaque_ap.get()));
    }
    return *this;
}

uint32_t
SBSymbolContextList::GetSize() const
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetSize();
    return 0;
}

SBSymbolContext
SBSymbolContextList::GetContextAtIndex (uint32_t idx)
{
    SBSymbolContext sb_sc;
    if (m_opaque_ap.get())
    {
        SymbolContext sc;
        if (m_opaque_ap->GetContextAtIndex (idx, sc))
        {
            sb_sc.SetSymbolContext(&sc);
        }
    }
    return sb_sc;
}


bool
SBSymbolContextList::IsValid () const
{
    return m_opaque_ap.get() != NULL;
}



lldb_private::SymbolContextList*
SBSymbolContextList::operator->() const
{
    return m_opaque_ap.get();
}


lldb_private::SymbolContextList&
SBSymbolContextList::operator*() const
{
    assert (m_opaque_ap.get());
    return *m_opaque_ap.get();
}




