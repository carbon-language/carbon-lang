//===-- SBBroadcaster.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Log.h"
#include "lldb/lldb-forward-rtti.h"

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBEvent.h"

using namespace lldb;
using namespace lldb_private;


SBBroadcaster::SBBroadcaster () :
    m_opaque (NULL),
    m_opaque_owned (false)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBBroadcastetr::SBBroadcaster () ==> this = %p", this);
}


SBBroadcaster::SBBroadcaster (const char *name) :
    m_opaque (new Broadcaster (name)),
    m_opaque_owned (true)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBBroadcaster::SBBroadcaster (const char *name) name = '%s' ==> this = %p (m_opaque = %p)",
                     name, this, m_opaque);
}

SBBroadcaster::SBBroadcaster (lldb_private::Broadcaster *broadcaster, bool owns) :
    m_opaque (broadcaster),
    m_opaque_owned (owns)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBBroadcaster::SBBroadcaster (lldb_private::Broadcaster *broadcaster, bool owns) "
                     " broadcaster = %p, owns = %s ==> this = %p", broadcaster, (owns ? "true" : "false"), this);
}

SBBroadcaster::~SBBroadcaster()
{
    reset (NULL, false);
}

void
SBBroadcaster::BroadcastEventByType (uint32_t event_type, bool unique)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBBroadcaster::BroadcastEventByType (%d, %s)", event_type, (unique ? "true" : "false"));

    if (m_opaque == NULL)
        return;

    if (unique)
        m_opaque->BroadcastEventIfUnique (event_type);
    else
        m_opaque->BroadcastEvent (event_type);
}

void
SBBroadcaster::BroadcastEvent (const SBEvent &event, bool unique)
{
    if (m_opaque == NULL)
        return;

    EventSP event_sp = event.GetSP ();
    if (unique)
        m_opaque->BroadcastEventIfUnique (event_sp);
    else
        m_opaque->BroadcastEvent (event_sp);
}

void
SBBroadcaster::AddInitialEventsToListener (const SBListener &listener, uint32_t requested_events)
{
    if (m_opaque)
        m_opaque->AddInitialEventsToListener (listener.get(), requested_events);
}

uint32_t
SBBroadcaster::AddListener (const SBListener &listener, uint32_t event_mask)
{
    if (m_opaque)
        return m_opaque->AddListener (listener.get(), event_mask);
    return 0;
}

const char *
SBBroadcaster::GetName ()
{
    if (m_opaque)
        return m_opaque->GetBroadcasterName().AsCString();
    return NULL;
}

bool
SBBroadcaster::EventTypeHasListeners (uint32_t event_type)
{
    if (m_opaque)
        return m_opaque->EventTypeHasListeners (event_type);
    return false;
}

bool
SBBroadcaster::RemoveListener (const SBListener &listener, uint32_t event_mask)
{
    if (m_opaque)
        return m_opaque->RemoveListener (listener.get(), event_mask);
    return false;
}

Broadcaster *
SBBroadcaster::get () const
{
    return m_opaque;
}

void
SBBroadcaster::reset (Broadcaster *broadcaster, bool owns)
{
    if (m_opaque && m_opaque_owned)
        delete m_opaque;
    m_opaque = broadcaster;
    m_opaque_owned = owns;
}


bool
SBBroadcaster::IsValid () const
{
    return m_opaque != NULL;
}

bool
SBBroadcaster::operator == (const SBBroadcaster &rhs) const
{
    return m_opaque == rhs.m_opaque;
    
}

bool
SBBroadcaster::operator != (const SBBroadcaster &rhs) const
{
    return m_opaque != rhs.m_opaque;
}
