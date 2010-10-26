//===-- SBEvent.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBEvent.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBStream.h"

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
    m_opaque (NULL)
{
}

SBEvent::SBEvent (uint32_t event_type, const char *cstr, uint32_t cstr_len) :
    m_event_sp (new Event (event_type, new EventDataBytes (cstr, cstr_len))),
    m_opaque (m_event_sp.get())
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
    {
        log->Printf ("SBEvent::SBEvent (event_type=%d, cstr='%s', cstr_len=%d)  => this.sp = %p", event_type,
                     cstr, cstr_len, m_opaque);
    }
}

SBEvent::SBEvent (EventSP &event_sp) :
    m_event_sp (event_sp),
    m_opaque (event_sp.get())
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBEvent::SBEvent (event_sp=%p) => this.sp = %p", event_sp.get(), m_opaque);
}

SBEvent::~SBEvent()
{
}

const char *
SBEvent::GetDataFlavor ()
{
    Event *lldb_event = get();
    if (lldb_event)
        return lldb_event->GetData()->GetFlavor().AsCString();
    return NULL;
}

uint32_t
SBEvent::GetType () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBEvent::GetType ()");

    const Event *lldb_event = get();
    uint32_t event_type = 0;
    if (lldb_event)
        event_type = lldb_event->GetType();

    if (log)
        log->Printf ("SBEvent::GetType (this.sp=%p) => %d", m_opaque, event_type);

    return event_type;
}

SBBroadcaster
SBEvent::GetBroadcaster () const
{
    SBBroadcaster broadcaster;
    const Event *lldb_event = get();
    if (lldb_event)
        broadcaster.reset (lldb_event->GetBroadcaster(), false);
    return broadcaster;
}

bool
SBEvent::BroadcasterMatchesPtr (const SBBroadcaster *broadcaster)
{
    if (broadcaster)
    {
        Event *lldb_event = get();
        if (lldb_event)
            return lldb_event->BroadcasterIs (broadcaster->get());
    }
    return false;
}

bool
SBEvent::BroadcasterMatchesRef (const SBBroadcaster &broadcaster)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBEvent::BroadcasterMatchesRef (broacaster) broadcaster = %p", &broadcaster);

    Event *lldb_event = get();
    bool success = false;
    if (lldb_event)
        success = lldb_event->BroadcasterIs (broadcaster.get());

    if (log)
        log->Printf ("SBEvent::BroadcasterMathesRef (this.sp=%p, broadcaster.obj=%p) => %s", m_opaque, 
                     broadcaster.get(), (success ? "true" : "false"));

    return success;
}

void
SBEvent::Clear()
{
    Event *lldb_event = get();
    if (lldb_event)
        lldb_event->Clear();
}

EventSP &
SBEvent::GetSP () const
{
    return m_event_sp;
}

Event *
SBEvent::get() const
{
    // There is a dangerous accessor call GetSharedPtr which can be used, so if
    // we have anything valid in m_event_sp, we must use that since if it gets
    // used by a function that puts something in there, then it won't update
    // m_opaque...
    if (m_event_sp)
        m_opaque = m_event_sp.get();

    return m_opaque;
}

void
SBEvent::reset (EventSP &event_sp)
{
    m_event_sp = event_sp;
    m_opaque = m_event_sp.get();
}

void
SBEvent::reset (Event* event_ptr)
{
    m_opaque = event_ptr;
    m_event_sp.reset();
}

bool
SBEvent::IsValid() const
{
    // Do NOT use m_opaque directly!!! Must use the SBEvent::get()
    // accessor. See comments in SBEvent::get()....
    return SBEvent::get() != NULL;

}

const char *
SBEvent::GetCStringFromEvent (const SBEvent &event)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("GetCStringFromEvent (event.sp=%p) => %s", event.m_opaque, 
                     reinterpret_cast<const char *>(EventDataBytes::GetBytesFromEvent (event.get())));

    return reinterpret_cast<const char *>(EventDataBytes::GetBytesFromEvent (event.get()));
}


bool
SBEvent::GetDescription (SBStream &description)
{
    if (m_opaque)
    {
        description.ref();
        m_opaque->Dump (description.get());
    }
    else
        description.Printf ("No value");

    return true;
}

bool
SBEvent::GetDescription (SBStream &description) const
{
    if (m_opaque)
    {
        description.ref();
        m_opaque->Dump (description.get());
    }
    else
        description.Printf ("No value");

    return true;
}
