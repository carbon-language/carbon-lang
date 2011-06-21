//===-- SBSymbolContextList.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSymbolContextList_h_
#define LLDB_SBSymbolContextList_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBSymbolContext.h"

namespace lldb {

class SBSymbolContextList
{
public:
    SBSymbolContextList ();

    SBSymbolContextList (const lldb::SBSymbolContextList& rhs);

    ~SBSymbolContextList ();

#ifndef SWIG
    const lldb::SBSymbolContextList &
    operator = (const lldb::SBSymbolContextList &rhs);
#endif

    bool
    IsValid () const;

    uint32_t
    GetSize() const;

    SBSymbolContext
    GetContextAtIndex (uint32_t idx);

    void
    Clear();

protected:

    friend class SBModule;
    friend class SBTarget;

#ifndef SWIG

    lldb_private::SymbolContextList*
    operator->() const;

    lldb_private::SymbolContextList&
    operator*() const;

#endif

private:
    std::auto_ptr<lldb_private::SymbolContextList> m_opaque_ap;
};


} // namespace lldb

#endif // LLDB_SBSymbolContextList_h_
