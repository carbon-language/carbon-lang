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

#include <LLDB/SBDefines.h>
#include <LLDB/SBBreakpoint.h>

namespace lldb {

class SBBreakpointLocation
{
public:

    SBBreakpointLocation ();

    ~SBBreakpointLocation ();

    bool
    IsValid() const;

    lldb::addr_t
    GetLoadAddress ();

    void
    SetEnabled(bool enabled);

    bool
    IsEnabled ();

    int32_t
    GetIgnoreCount ();

    void
    SetIgnoreCount (int32_t n);

    void
    SetThreadID (lldb::tid_t thread_id);

    lldb::tid_t
    GetThreadID ();

    bool
    IsResolved ();

    void
    GetDescription (FILE *f, const char *description_level);

    SBBreakpoint
    GetBreakpoint ();

private:
    friend class SBBreakpoint;

    SBBreakpointLocation (const lldb::BreakpointLocationSP &break_loc_sp);
    
    void
    SetLocation (const lldb::BreakpointLocationSP &break_loc_sp);

    lldb::BreakpointLocationSP m_break_loc_sp;

};

} // namespace lldb

#endif  // LLDB_SBBreakpointLocation_h_
