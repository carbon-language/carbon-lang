//===-- SBWatchpoint.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBWatchpoint.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBStream.h"

#include "lldb/lldb-types.h"
#include "lldb/lldb-defines.h"
#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Breakpoint/WatchpointList.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;


SBWatchpoint::SBWatchpoint () :
    m_opaque_sp ()
{
}

SBWatchpoint::SBWatchpoint (const lldb::WatchpointSP &wp_sp) :
    m_opaque_sp (wp_sp)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
    {
        SBStream sstr;
        GetDescription (sstr, lldb::eDescriptionLevelBrief);
        log->Printf ("SBWatchpoint::SBWatchpoint (const lldb::WatchpointSP &wp_sp"
                     "=%p)  => this.sp = %p (%s)", wp_sp.get(), m_opaque_sp.get(), sstr.GetData());
    }
}

SBWatchpoint::SBWatchpoint(const SBWatchpoint &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

const SBWatchpoint &
SBWatchpoint::operator = (const SBWatchpoint &rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
}


SBWatchpoint::~SBWatchpoint ()
{
}

watch_id_t
SBWatchpoint::GetID ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    watch_id_t watch_id = LLDB_INVALID_WATCH_ID;
    if (m_opaque_sp)
        watch_id = m_opaque_sp->GetID();

    if (log)
    {
        if (watch_id == LLDB_INVALID_WATCH_ID)
            log->Printf ("SBWatchpoint(%p)::GetID () => LLDB_INVALID_WATCH_ID", m_opaque_sp.get());
        else
            log->Printf ("SBWatchpoint(%p)::GetID () => %u", m_opaque_sp.get(), watch_id);
    }

    return watch_id;
}

bool
SBWatchpoint::IsValid() const
{
    return m_opaque_sp.get() != NULL;
#if 0
    if (m_opaque_sp)
        return m_opaque_sp->GetError().Success();
    return false;
#endif
}

SBError
SBWatchpoint::GetError ()
{
    SBError sb_error;
#if 0
    if (m_opaque_sp)
    {
        // TODO: Johnny fill this in
        sb_error.ref() = m_opaque_sp->GetError();
    }
#endif
    return sb_error;
}

int32_t
SBWatchpoint::GetHardwareIndex ()
{
    int32_t hw_index = -1;

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        hw_index = m_opaque_sp->GetHardwareIndex();
    }

    return hw_index;
}

addr_t
SBWatchpoint::GetWatchAddress ()
{
    addr_t ret_addr = LLDB_INVALID_ADDRESS;

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        ret_addr = m_opaque_sp->GetLoadAddress();
    }

    return ret_addr;
}

size_t
SBWatchpoint::GetWatchSize ()
{
    size_t watch_size = 0;

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        watch_size = m_opaque_sp->GetByteSize();
    }

    return watch_size;
}

void
SBWatchpoint::SetEnabled (bool enabled)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        m_opaque_sp->GetTarget().DisableWatchpointByID(m_opaque_sp->GetID());
    }
}

bool
SBWatchpoint::IsEnabled ()
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        return m_opaque_sp->IsEnabled();
    }
    else
        return false;
}

uint32_t
SBWatchpoint::GetHitCount ()
{
    uint32_t count = 0;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        count = m_opaque_sp->GetHitCount();
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBWatchpoint(%p)::GetHitCount () => %u", m_opaque_sp.get(), count);

    return count;
}

uint32_t
SBWatchpoint::GetIgnoreCount ()
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        return m_opaque_sp->GetIgnoreCount();
    }
    else
        return 0;
}

void
SBWatchpoint::SetIgnoreCount (uint32_t n)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        m_opaque_sp->SetIgnoreCount (n);
    }
}

bool
SBWatchpoint::GetDescription (SBStream &description, DescriptionLevel level)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        description.ref();
        m_opaque_sp->GetDescription (description.get(), level);
        description.get()->EOL();
    }
    else
        description.Printf ("No value");

    return true;
}

lldb_private::Watchpoint *
SBWatchpoint::operator->()
{
    return m_opaque_sp.get();
}

lldb_private::Watchpoint *
SBWatchpoint::get()
{
    return m_opaque_sp.get();
}

lldb::WatchpointSP &
SBWatchpoint::operator *()
{
    return m_opaque_sp;
}
