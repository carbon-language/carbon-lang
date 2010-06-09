//===-- SBSymbolContext.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSymbolContext_h_
#define LLDB_SBSymbolContext_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBBlock.h"
#include "lldb/API/SBCompileUnit.h"
#include "lldb/API/SBFunction.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBSymbol.h"

namespace lldb {

class SBSymbolContext
{
public:
    SBSymbolContext ();

    SBSymbolContext (const lldb::SBSymbolContext& rhs);

    ~SBSymbolContext ();

    bool
    IsValid () const;

#ifndef SWIG
    const lldb::SBSymbolContext &
    operator = (const lldb::SBSymbolContext &rhs);
#endif

    SBModule        GetModule ();
    SBCompileUnit   GetCompileUnit ();
    SBFunction      GetFunction ();
    SBBlock         GetBlock ();
    SBLineEntry     GetLineEntry ();
    SBSymbol        GetSymbol ();

protected:
    friend class SBFrame;
    friend class SBThread;

#ifndef SWIG

    lldb_private::SymbolContext*
    operator->() const;

#endif

    lldb_private::SymbolContext *
    GetLLDBObjectPtr() const;

    SBSymbolContext (const lldb_private::SymbolContext *sc_ptr);

    void
    SetSymbolContext (const lldb_private::SymbolContext *sc_ptr);

private:
    std::auto_ptr<lldb_private::SymbolContext> m_lldb_object_ap;
};


} // namespace lldb

#endif // LLDB_SBSymbolContext_h_
