//===-- SBBroadcaster.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Broadcaster.h"
#include "lldb/lldb-forward-rtti.h"

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBEvent.h"

using namespace lldb;
using namespace lldb_private;


SBBroadcaster::SBBroadcaster () :
    m_lldb_object (NULL),
    m_lldb_object_owned (false)
{
}


SBBroadcaster::SBBroadcaster (const char *name) :
    m_lldb_object (new Broadcaster (name)),
    m_lldb_object_owned (true)
{
}

SBBroadcaster::SBBroadcaster (lldb_private::Broadcaster *broadcaster, bool owns) :
    m_lldb_object (broadcaster),
    m_lldb_object_owned (owns)
{
}

SBBroadcaster::~SBBroadcaster()
{
    SetLLDBObjectPtr (NULL, false);
}

void
SBBroadcaster::BroadcastEventByType (uint32_t event_type, bool unique)
{
    if (m_lldb_object == NULL)
        return;

    if (unique)
        m_lldb_object->BroadcastEventIfUnique (event_type);
    else
        m_lldb_object->BroadcastEvent (event_type);
}

void
SBBroadcaster::BroadcastEvent (const SBEvent &event, bool unique)
{
    if (m_lldb_object == NULL)
        return;

    EventSP event_sp = event.GetSharedPtr ();
    if (unique)
        m_lldb_object->BroadcastEventIfUnique (event_sp);
    else
        m_lldb_object->BroadcastEvent (event_sp);
}

void
SBBroadcaster::AddInitialEventsToListener (const SBListener &listener, uint32_t requested_events)
{
    if (m_lldb_object)
        m_lldb_object->AddInitialEventsToListener (listener.get(), requested_events);
}

uint32_t
SBBroadcaster::AddListener (const SBListener &listener, uint32_t event_mask)
{
    if (m_lldb_object)
        return m_lldb_object->AddListener (listener.get(), event_mask);
    return 0;
}

const char *
SBBroadcaster::GetName ()
{
    if (m_lldb_object)
        return m_lldb_object->GetBroadcasterName().AsCString();
    return NULL;
}

bool
SBBroadcaster::EventTypeHasListeners (uint32_t event_type)
{
    if (m_lldb_object)
        return m_lldb_object->EventTypeHasListeners (event_type);
    return false;
}

bool
SBBroadcaster::RemoveListener (const SBListener &listener, uint32_t event_mask)
{
    if (m_lldb_object)
        return m_lldb_object->RemoveListener (listener.get(), event_mask);
    return false;
}

Broadcaster *
SBBroadcaster::GetLLDBObjectPtr () const
{
    return m_lldb_object;
}

void
SBBroadcaster::SetLLDBObjectPtr (Broadcaster *broadcaster, bool owns)
{
    if (m_lldb_object && m_lldb_object_owned)
        delete m_lldb_object;
    m_lldb_object = broadcaster;
    m_lldb_object_owned = owns;
}


bool
SBBroadcaster::IsValid () const
{
    return m_lldb_object != NULL;
}

bool
SBBroadcaster::operator == (const SBBroadcaster &rhs) const
{
    return m_lldb_object == rhs.m_lldb_object;
    
}

bool
SBBroadcaster::operator != (const SBBroadcaster &rhs) const
{
    return m_lldb_object != rhs.m_lldb_object;
}
