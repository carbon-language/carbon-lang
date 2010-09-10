//===-- SBSymbolContext.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSymbolContext.h"
#include "lldb/Symbol/SymbolContext.h"

using namespace lldb;
using namespace lldb_private;



SBSymbolContext::SBSymbolContext () :
    m_opaque_ap ()
{
}

SBSymbolContext::SBSymbolContext (const SymbolContext *sc_ptr) :
    m_opaque_ap ()
{
    if (sc_ptr)
        m_opaque_ap.reset (new SymbolContext (*sc_ptr));
}

SBSymbolContext::SBSymbolContext (const SBSymbolContext& rhs) :
    m_opaque_ap ()
{
    if (rhs.IsValid())
        *m_opaque_ap = *rhs.m_opaque_ap;
}

SBSymbolContext::~SBSymbolContext ()
{
}

const SBSymbolContext &
SBSymbolContext::operator = (const SBSymbolContext &rhs)
{
    if (this != &rhs)
    {
        if (rhs.IsValid())
            m_opaque_ap.reset (new lldb_private::SymbolContext(*rhs.m_opaque_ap.get()));
    }
    return *this;
}

void
SBSymbolContext::SetSymbolContext (const SymbolContext *sc_ptr)
{
    if (sc_ptr)
    {
        if (m_opaque_ap.get())
            *m_opaque_ap = *sc_ptr;
        else
            m_opaque_ap.reset (new SymbolContext (*sc_ptr));
    }
    else
    {
        if (m_opaque_ap.get())
            m_opaque_ap->Clear();
    }
}

bool
SBSymbolContext::IsValid () const
{
    return m_opaque_ap.get() != NULL;
}



SBModule
SBSymbolContext::GetModule ()
{
    SBModule sb_module;
    if (m_opaque_ap.get())
        sb_module.SetModule(m_opaque_ap->module_sp);
    return sb_module;
}

SBCompileUnit
SBSymbolContext::GetCompileUnit ()
{
    return SBCompileUnit (m_opaque_ap.get() ? m_opaque_ap->comp_unit : NULL);
}

SBFunction
SBSymbolContext::GetFunction ()
{
    return SBFunction (m_opaque_ap.get() ? m_opaque_ap->function : NULL);
}

SBBlock
SBSymbolContext::GetBlock ()
{
    return SBBlock (m_opaque_ap.get() ? m_opaque_ap->block : NULL);
}

SBLineEntry
SBSymbolContext::GetLineEntry ()
{
    SBLineEntry sb_line_entry;
    if (m_opaque_ap.get())
        sb_line_entry.SetLineEntry (m_opaque_ap->line_entry);

    return sb_line_entry;
}

SBSymbol
SBSymbolContext::GetSymbol ()
{
    return SBSymbol (m_opaque_ap.get() ? m_opaque_ap->symbol : NULL);
}

lldb_private::SymbolContext*
SBSymbolContext::operator->() const
{
    return m_opaque_ap.get();
}


const lldb_private::SymbolContext&
SBSymbolContext::operator*() const
{
    assert (m_opaque_ap.get());
    return *m_opaque_ap.get();
}


lldb_private::SymbolContext&
SBSymbolContext::operator*()
{
    if (m_opaque_ap.get() == NULL)
        m_opaque_ap.reset (new SymbolContext);
    return *m_opaque_ap.get();
}

lldb_private::SymbolContext *
SBSymbolContext::get() const
{
    return m_opaque_ap.get();
}



