//===-- SBBreakpoint.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBreakpoint_h_
#define LLDB_SBBreakpoint_h_

#include "lldb/API/SBDefines.h"
#include <stdio.h>

namespace lldb {

class SBBreakpoint
{
public:

    typedef bool (*BreakpointHitCallback) (void *baton, 
                                           SBProcess &process,
                                           SBThread &thread, 
                                           lldb::SBBreakpointLocation &location);

    SBBreakpoint ();

    SBBreakpoint (const lldb::SBBreakpoint& rhs);

    ~SBBreakpoint();

#ifndef SWIG
    const SBBreakpoint &
    operator = (const SBBreakpoint& rhs);
#endif

    break_id_t
    GetID () const;

    bool
    IsValid() const;

    void
    Dump (FILE *f);

    void
    ClearAllBreakpointSites ();

    lldb::SBBreakpointLocation
    FindLocationByAddress (lldb::addr_t vm_addr);

    lldb::break_id_t
    FindLocationIDByAddress (lldb::addr_t vm_addr);

    lldb::SBBreakpointLocation
    FindLocationByID (lldb::break_id_t bp_loc_id);

    lldb::SBBreakpointLocation
    GetLocationAtIndex (uint32_t index);

    void
    ListLocations (FILE *, const char *description_level = "full");

    void
    SetEnabled (bool enable);

    bool
    IsEnabled ();

    void
    SetIgnoreCount (int32_t count);

    int32_t
    GetIgnoreCount () const;

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

    void
    SetCallback (BreakpointHitCallback callback, void *baton);

    size_t
    GetNumResolvedLocations() const;

    size_t
    GetNumLocations() const;

    void
    GetDescription (FILE *, const char *description_level, bool describe_locations = false);



private:
    friend class SBBreakpointLocation;
    friend class SBTarget;

    SBBreakpoint (const lldb::BreakpointSP &bp_sp);

#ifndef SWIG

    lldb_private::Breakpoint *
    operator->() const;

    lldb_private::Breakpoint *
    get() const;

    lldb::BreakpointSP &
    operator *();

    const lldb::BreakpointSP &
    operator *() const;

#endif

    static bool
    PrivateBreakpointHitCallback (void *baton, 
                                  lldb_private::StoppointCallbackContext *context, 
                                  lldb::user_id_t break_id, 
                                  lldb::user_id_t break_loc_id);
    
    lldb::BreakpointSP m_break_sp;
};

} // namespace lldb

#endif  // LLDB_SBBreakpoint_h_
