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

#ifdef SWIG
%feature("docstring",
"A context object that provides access to core debugger entities.

Manay debugger functions require a context when doing lookups. This class
provides a common structure that can be used as the result of a query that
can contain a single result.

For example,

        exe = os.path.join(os.getcwd(), 'a.out')

        # Create a target for the debugger.
        target = self.dbg.CreateTarget(exe)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        # The inferior should stop on 'c'.
        from lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        # Now get the SBSymbolContext from this frame.  We want everything. :-)
        context = frame0.GetSymbolContext(lldb.eSymbolContextEverything)

        # Get the module.
        module = context.GetModule()
        ...

        # And the compile unit associated with the frame.
        compileUnit = context.GetCompileUnit()
        ...
"
         ) SBSymbolContext;
#endif
class SBSymbolContext
{
#ifdef SWIG
    %feature("autodoc", "1");
#endif
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

    bool
    GetDescription (lldb::SBStream &description);

protected:
    friend class SBFrame;
    friend class SBModule;
    friend class SBThread;
    friend class SBTarget;
    friend class SBSymbolContextList;

#ifndef SWIG

    lldb_private::SymbolContext*
    operator->() const;

    lldb_private::SymbolContext&
    operator*();

    lldb_private::SymbolContext&
    ref();

    const lldb_private::SymbolContext&
    operator*() const;

#endif

    lldb_private::SymbolContext *
    get() const;

    SBSymbolContext (const lldb_private::SymbolContext *sc_ptr);

    void
    SetSymbolContext (const lldb_private::SymbolContext *sc_ptr);

private:
    std::auto_ptr<lldb_private::SymbolContext> m_opaque_ap;
};


} // namespace lldb

#endif // LLDB_SBSymbolContext_h_
