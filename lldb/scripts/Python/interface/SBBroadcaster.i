//===-- SWIG Interface for SBBroadcaster ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBBroadcaster
{
public:
    SBBroadcaster ();

    SBBroadcaster (const char *name);

    SBBroadcaster (const SBBroadcaster &rhs);
    
    ~SBBroadcaster();

    bool
    IsValid () const;

    void
    Clear ();

    void
    BroadcastEventByType (uint32_t event_type, bool unique = false);

    void
    BroadcastEvent (const lldb::SBEvent &event, bool unique = false);

    void
    AddInitialEventsToListener (const lldb::SBListener &listener, uint32_t requested_events);

    uint32_t
    AddListener (const lldb::SBListener &listener, uint32_t event_mask);

    const char *
    GetName () const;

    bool
    EventTypeHasListeners (uint32_t event_type);

    bool
    RemoveListener (const lldb::SBListener &listener, uint32_t event_mask = UINT32_MAX);
};

} // namespace lldb
