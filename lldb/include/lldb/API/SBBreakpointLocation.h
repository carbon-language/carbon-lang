//===-- SBBreakpointLocation.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBreakpointLocation_h_
#define LLDB_SBBreakpointLocation_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBBreakpoint.h"

namespace lldb {

#ifdef SWIG
%feature("docstring",
"Represents one unique instance (by address) of a logical breakpoint.

A breakpoint location is defined by the breakpoint that produces it,
and the address that resulted in this particular instantiation.
Each breakpoint location has its settable options.

SBBreakpoint contains SBBreakpointLocation(s). See docstring of SBBreakpoint
for retrieval of an SBBreakpointLocation from an SBBreakpoint.
"
         ) SBBreakpointLocation;
#endif
class SBBreakpointLocation
{
#ifdef SWIG
    %feature("autodoc", "1");
#endif
public:

    SBBreakpointLocation ();

    SBBreakpointLocation (const lldb::SBBreakpointLocation &rhs);

    ~SBBreakpointLocation ();

#ifndef SWIG
    const lldb::SBBreakpointLocation &
    operator = (const lldb::SBBreakpointLocation &rhs);
#endif

    bool
    IsValid() const;

    lldb::addr_t
    GetLoadAddress ();

    void
    SetEnabled(bool enabled);

    bool
    IsEnabled ();

    uint32_t
    GetIgnoreCount ();

    void
    SetIgnoreCount (uint32_t n);

    void 
    SetCondition (const char *condition);
    
    const char *
    GetCondition ();

    void
    SetThreadID (lldb::tid_t sb_thread_id);

    lldb::tid_t
    GetThreadID ();
    
    void
    SetThreadIndex (uint32_t index);
    
    uint32_t
    GetThreadIndex() const;
    
    void
    SetThreadName (const char *thread_name);
    
    const char *
    GetThreadName () const;
    
    void 
    SetQueueName (const char *queue_name);
    
    const char *
    GetQueueName () const;

    bool
    IsResolved ();

    bool
    GetDescription (lldb::SBStream &description, DescriptionLevel level);

    SBBreakpoint
    GetBreakpoint ();

#ifndef SWIG
    SBBreakpointLocation (const lldb::BreakpointLocationSP &break_loc_sp);
#endif

private:
    friend class SBBreakpoint;
    friend class lldb_private::ScriptInterpreterPython;
    
    void
    SetLocation (const lldb::BreakpointLocationSP &break_loc_sp);

    lldb::BreakpointLocationSP m_opaque_sp;

};

} // namespace lldb

#endif  // LLDB_SBBreakpointLocation_h_
