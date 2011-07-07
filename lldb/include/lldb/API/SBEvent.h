//===-- SBEvent.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBEvent_h_
#define LLDB_SBEvent_h_

#include "lldb/API/SBDefines.h"

#include <stdio.h>
#include <vector>


namespace lldb {

class SBBroadcaster;

class SBEvent
{
public:
    SBEvent();

    // Make an event that contains a C string.
    SBEvent (uint32_t event, const char *cstr, uint32_t cstr_len);

    ~SBEvent();

    SBEvent (const lldb::SBEvent &rhs);
    
#ifndef SWIG
    const SBEvent &
    operator = (const lldb::SBEvent &rhs);
#endif

    bool
    IsValid() const;

    const char *
    GetDataFlavor ();

    uint32_t
    GetType () const;

    lldb::SBBroadcaster
    GetBroadcaster () const;

#ifndef SWIG
    bool
    BroadcasterMatchesPtr (const lldb::SBBroadcaster *broadcaster);
#endif

    bool
    BroadcasterMatchesRef (const lldb::SBBroadcaster &broadcaster);

    void
    Clear();

    static const char *
    GetCStringFromEvent (const lldb::SBEvent &event);

#ifndef SWIG
    bool
    GetDescription (lldb::SBStream &description);
#endif

    bool
    GetDescription (lldb::SBStream &description) const;

protected:
    friend class SBListener;
    friend class SBBroadcaster;
    friend class SBBreakpoint;
    friend class SBDebugger;
    friend class SBProcess;

    SBEvent (lldb::EventSP &event_sp);

#ifndef SWIG

    lldb::EventSP &
    GetSP () const;

    void
    reset (lldb::EventSP &event_sp);

    void
    reset (lldb_private::Event* event);

    lldb_private::Event *
    get () const;

#endif

private:

    mutable lldb::EventSP m_event_sp;
    mutable lldb_private::Event *m_opaque_ptr;
};

} // namespace lldb

#endif  // LLDB_SBEvent_h_
