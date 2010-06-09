//===-- SBListener.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Listener.h"
#include "lldb/lldb-forward-rtti.h"
#include "lldb/Host/TimeValue.h"

#include "lldb/API/SBListener.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBBroadcaster.h"

using namespace lldb;
using namespace lldb_private;


SBListener::SBListener ()
{
}

SBListener::SBListener (const char *name) :
    m_lldb_object_ptr (new Listener (name)),
    m_lldb_object_ptr_owned (true)
{
}

SBListener::SBListener (Listener &listener) :
    m_lldb_object_ptr (&listener),
    m_lldb_object_ptr_owned (false)
{
}

SBListener::~SBListener ()
{
    if (m_lldb_object_ptr_owned)
    {
        if (m_lldb_object_ptr)
        {
            delete m_lldb_object_ptr;
            m_lldb_object_ptr = NULL;
        }
    }
}

bool
SBListener::IsValid() const
{
    return m_lldb_object_ptr != NULL;
}

void
SBListener::AddEvent (const SBEvent &event)
{
    EventSP &event_sp = event.GetSharedPtr ();
    if (event_sp)
        m_lldb_object_ptr->AddEvent (event_sp);
}

void
SBListener::Clear ()
{
    if (IsValid())
        m_lldb_object_ptr->Clear ();
}

uint32_t
SBListener::StartListeningForEvents (const SBBroadcaster& broadcaster, uint32_t event_mask)
{
    if (IsValid() && broadcaster.IsValid())
    {
        return m_lldb_object_ptr->StartListeningForEvents (broadcaster.GetLLDBObjectPtr (), event_mask);
    }
    return false;
}

bool
SBListener::StopListeningForEvents (const SBBroadcaster& broadcaster, uint32_t event_mask)
{
    if (IsValid() && broadcaster.IsValid())
    {
        return m_lldb_object_ptr->StopListeningForEvents (broadcaster.GetLLDBObjectPtr (), event_mask);
    }
    return false;
}

bool
SBListener::WaitForEvent (uint32_t num_seconds, SBEvent &event)
{
    if (IsValid())
    {
        TimeValue time_value;
        if (num_seconds != UINT32_MAX)
        {
            assert (num_seconds != 0); // Take this out after all calls with timeout set to zero have been removed....
            time_value = TimeValue::Now();
            time_value.OffsetWithSeconds (num_seconds);
        }
        EventSP event_sp;
        if (m_lldb_object_ptr->WaitForEvent (time_value.IsValid() ? &time_value : NULL, event_sp))
        {
            event.SetEventSP (event_sp);
            return true;
        }
    }
    event.SetLLDBObjectPtr (NULL);
    return false;
}

bool
SBListener::WaitForEventForBroadcaster
(
    uint32_t num_seconds,
    const SBBroadcaster &broadcaster,
    SBEvent &event
)
{
    if (IsValid() && broadcaster.IsValid())
    {
        TimeValue time_value;
        if (num_seconds != UINT32_MAX)
        {
            time_value = TimeValue::Now();
            time_value.OffsetWithSeconds (num_seconds);
        }
        EventSP event_sp;
        if (m_lldb_object_ptr->WaitForEventForBroadcaster (time_value.IsValid() ? &time_value : NULL,
                                                         broadcaster.GetLLDBObjectPtr (),
                                                         event_sp))
        {
            event.SetEventSP (event_sp);
            return true;
        }

    }
    event.SetLLDBObjectPtr (NULL);
    return false;
}

bool
SBListener::WaitForEventForBroadcasterWithType
(
    uint32_t num_seconds,
    const SBBroadcaster &broadcaster,
    uint32_t event_type_mask,
    SBEvent &event
)
{
    if (IsValid() && broadcaster.IsValid())
    {
        TimeValue time_value;
        if (num_seconds != UINT32_MAX)
        {
            time_value = TimeValue::Now();
            time_value.OffsetWithSeconds (num_seconds);
        }
        EventSP event_sp;
        if (m_lldb_object_ptr->WaitForEventForBroadcasterWithType (time_value.IsValid() ? &time_value : NULL,
                                                                 broadcaster.GetLLDBObjectPtr (),
                                                                 event_type_mask,
                                                                 event_sp))
        {
            event.SetEventSP (event_sp);
            return true;
        }
    }
    event.SetLLDBObjectPtr (NULL);
    return false;
}

bool
SBListener::PeekAtNextEvent (SBEvent &event)
{
    if (m_lldb_object_ptr)
    {
        event.SetLLDBObjectPtr (m_lldb_object_ptr->PeekAtNextEvent ());
        return event.IsValid();
    }
    event.SetLLDBObjectPtr (NULL);
    return false;
}

bool
SBListener::PeekAtNextEventForBroadcaster (const SBBroadcaster &broadcaster, SBEvent &event)
{
    if (IsValid() && broadcaster.IsValid())
    {
        event.SetLLDBObjectPtr (m_lldb_object_ptr->PeekAtNextEventForBroadcaster (broadcaster.GetLLDBObjectPtr ()));
        return event.IsValid();
    }
    event.SetLLDBObjectPtr (NULL);
    return false;
}

bool
SBListener::PeekAtNextEventForBroadcasterWithType (const SBBroadcaster &broadcaster, uint32_t event_type_mask,
                                                   SBEvent &event)
{
    if (IsValid() && broadcaster.IsValid())
    {
        event.SetLLDBObjectPtr(m_lldb_object_ptr->PeekAtNextEventForBroadcasterWithType (broadcaster.GetLLDBObjectPtr (), event_type_mask));
        return event.IsValid();
    }
    event.SetLLDBObjectPtr (NULL);
    return false;
}

bool
SBListener::GetNextEvent (SBEvent &event)
{
    if (m_lldb_object_ptr)
    {
        EventSP event_sp;
        if (m_lldb_object_ptr->GetNextEvent (event_sp))
        {
            event.SetEventSP (event_sp);
            return true;
        }
    }
    event.SetLLDBObjectPtr (NULL);
    return false;
}

bool
SBListener::GetNextEventForBroadcaster (const SBBroadcaster &broadcaster, SBEvent &event)
{
    if (IsValid() && broadcaster.IsValid())
    {
        EventSP event_sp;
        if (m_lldb_object_ptr->GetNextEventForBroadcaster (broadcaster.GetLLDBObjectPtr (), event_sp))
        {
            event.SetEventSP (event_sp);
            return true;
        }
    }
    event.SetLLDBObjectPtr (NULL);
    return false;
}

bool
SBListener::GetNextEventForBroadcasterWithType
(
    const SBBroadcaster &broadcaster,
    uint32_t event_type_mask,
    SBEvent &event
)
{
    if (IsValid() && broadcaster.IsValid())
    {
        EventSP event_sp;
        if (m_lldb_object_ptr->GetNextEventForBroadcasterWithType (broadcaster.GetLLDBObjectPtr (),
                                                                 event_type_mask,
                                                                 event_sp))
        {
            event.SetEventSP (event_sp);
            return true;
        }
    }
    event.SetLLDBObjectPtr (NULL);
    return false;
}

bool
SBListener::HandleBroadcastEvent (const SBEvent &event)
{
    if (m_lldb_object_ptr)
        return m_lldb_object_ptr->HandleBroadcastEvent (event.GetSharedPtr());
    return false;
}

lldb_private::Listener *
SBListener::operator->() const
{
    return m_lldb_object_ptr;
}

lldb_private::Listener *
SBListener::get() const
{
    return m_lldb_object_ptr;
}

lldb_private::Listener &
SBListener::operator *()
{
    return *m_lldb_object_ptr;
}

const lldb_private::Listener &
SBListener::operator *() const
{
    return *m_lldb_object_ptr;
}


