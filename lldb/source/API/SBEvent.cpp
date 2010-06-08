//===-- SBEvent.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SBEvent.h"
#include "SBBroadcaster.h"

#include "lldb/Core/Event.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Target/Process.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Interpreter/CommandInterpreter.h"

using namespace lldb;
using namespace lldb_private;


SBEvent::SBEvent () :
    m_event_sp (),
    m_lldb_object (NULL)
{
}

SBEvent::SBEvent (uint32_t event_type, const char *cstr, uint32_t cstr_len) :
    m_event_sp (new Event (event_type, new EventDataBytes (cstr, cstr_len))),
    m_lldb_object (m_event_sp.get())
{
}

SBEvent::SBEvent (EventSP &event_sp) :
    m_event_sp (event_sp),
    m_lldb_object (event_sp.get())
{
}

SBEvent::~SBEvent()
{
}

void
SBEvent::Dump (FILE *f) const
{
    const Event *lldb_event = GetLLDBObjectPtr();
    if (lldb_event)
    {
        StreamFile str(f);
        lldb_event->Dump ((Stream *) &str);
    }
}

const char *
SBEvent::GetDataFlavor ()
{
    Event *lldb_event = SBEvent::GetLLDBObjectPtr();
    if (lldb_event)
        return lldb_event->GetData()->GetFlavor().AsCString();
    return NULL;
}

uint32_t
SBEvent::GetType () const
{
    const Event *lldb_event = SBEvent::GetLLDBObjectPtr();
    if (lldb_event)
        return lldb_event->GetType();
    return 0;
}

SBBroadcaster
SBEvent::GetBroadcaster () const
{
    SBBroadcaster broadcaster;
    const Event *lldb_event = SBEvent::GetLLDBObjectPtr();
    if (lldb_event)
        broadcaster.SetLLDBObjectPtr (lldb_event->GetBroadcaster(), false);
    return broadcaster;
}

bool
SBEvent::BroadcasterMatchesPtr (const SBBroadcaster *broadcaster)
{
    if (broadcaster)
    {
        Event *lldb_event = SBEvent::GetLLDBObjectPtr();
        if (lldb_event)
            return lldb_event->BroadcasterIs (broadcaster->GetLLDBObjectPtr ());
    }
    return false;
}

bool
SBEvent::BroadcasterMatchesRef (const SBBroadcaster &broadcaster)
{
    Event *lldb_event = SBEvent::GetLLDBObjectPtr();
    if (lldb_event)
        return lldb_event->BroadcasterIs (broadcaster.GetLLDBObjectPtr ());
    return false;
}

void
SBEvent::Clear()
{
    Event *lldb_event = SBEvent::GetLLDBObjectPtr();
    if (lldb_event)
        lldb_event->Clear();
}

EventSP &
SBEvent::GetSharedPtr () const
{
    return m_event_sp;
}

Event *
SBEvent::GetLLDBObjectPtr ()
{
    // There is a dangerous accessor call GetSharedPtr which can be used, so if
    // we have anything valid in m_event_sp, we must use that since if it gets
    // used by a function that puts something in there, then it won't update
    // m_lldb_object...
    if (m_event_sp)
        m_lldb_object = m_event_sp.get();

    return m_lldb_object;
}

const Event *
SBEvent::GetLLDBObjectPtr () const
{
    // There is a dangerous accessor call GetSharedPtr which can be used, so if
    // we have anything valid in m_event_sp, we must use that since if it gets
    // used by a function that puts something in there, then it won't update
    // m_lldb_object...
    if (m_event_sp)
        m_lldb_object = m_event_sp.get();

    return m_lldb_object;
}

void
SBEvent::SetEventSP (EventSP &event_sp)
{
    m_event_sp = event_sp;
    m_lldb_object = m_event_sp.get();
}

void
SBEvent::SetLLDBObjectPtr (Event* event_ptr)
{
    m_lldb_object = event_ptr;
    m_event_sp.reset();
}

bool
SBEvent::IsValid() const
{
    // Do NOT use m_lldb_object directly!!! Must use the SBEvent::GetLLDBObjectPtr()
    // accessor. See comments in SBEvent::GetLLDBObjectPtr()....
    return SBEvent::GetLLDBObjectPtr() != NULL;

}

const char *
SBEvent::GetCStringFromEvent (const SBEvent &event)
{
    return reinterpret_cast<const char *>(EventDataBytes::GetBytesFromEvent (event.GetLLDBObjectPtr()));
}


