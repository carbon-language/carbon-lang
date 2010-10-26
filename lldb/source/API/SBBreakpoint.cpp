//===-- SBBreakpoint.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBThread.h"

#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"


#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;

struct CallbackData
{
    SBBreakpoint::BreakpointHitCallback callback;
    void *callback_baton;
};

class SBBreakpointCallbackBaton : public Baton
{
public:

    SBBreakpointCallbackBaton (SBBreakpoint::BreakpointHitCallback callback, void *baton) :
        Baton (new CallbackData)
    {
        CallbackData *data = (CallbackData *)m_data;
        data->callback = callback;
        data->callback_baton = baton;
    }
    
    virtual ~SBBreakpointCallbackBaton()
    {
        CallbackData *data = (CallbackData *)m_data;

        if (data)
        {
            delete data;
            m_data = NULL;
        }
    }
};


SBBreakpoint::SBBreakpoint () :
    m_opaque_sp ()
{
}

SBBreakpoint::SBBreakpoint (const SBBreakpoint& rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
    {
        SBStream sstr;
        GetDescription (sstr);
        log->Printf ("SBBreakpoint::SBBreakpoint (const SBBreakpoint rhs.sp=%p) "
                     "=> this.sp = %p (%s)",
                     rhs.m_opaque_sp.get(), m_opaque_sp.get(), sstr.GetData());
    }
}


SBBreakpoint::SBBreakpoint (const lldb::BreakpointSP &bp_sp) :
    m_opaque_sp (bp_sp)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
    {
        SBStream sstr;
        GetDescription (sstr);
        log->Printf("SBBreakpoint::SBBreakpoint (const lldb::BreakpointSP &bp_sp=%p) => this.sp = %p (%s)",
                    bp_sp.get(), m_opaque_sp.get(), sstr.GetData());
    }
}

SBBreakpoint::~SBBreakpoint()
{
}

const SBBreakpoint &
SBBreakpoint::operator = (const SBBreakpoint& rhs)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBBreakpoint::operator=");

    if (this != &rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
    }
    
    if (log)
        log->Printf ("SBBreakpoint::operator= (const SBBreakpoint &rhs.sp=%p) => this.sp = %p", 
                     rhs.m_opaque_sp.get(), m_opaque_sp.get());
        
    return *this;
}

break_id_t
SBBreakpoint::GetID () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBBreakpoint::GetID");

    if (m_opaque_sp)
    {
        break_id_t id = m_opaque_sp->GetID();
        if (log)
            log->Printf ("SBBreakpoint::GetID (this.sp=%p) => %d", m_opaque_sp.get(), id);
        return id;
    }

    if (log)
        log->Printf ("SBBreakpoint::GetID (this.sp=%p) => LLDB_INVALID_BREAK_ID", m_opaque_sp.get());

    return LLDB_INVALID_BREAK_ID;
}


bool
SBBreakpoint::IsValid() const
{
    return m_opaque_sp;
}

void
SBBreakpoint::ClearAllBreakpointSites ()
{
    if (m_opaque_sp)
        m_opaque_sp->ClearAllBreakpointSites ();
}

SBBreakpointLocation
SBBreakpoint::FindLocationByAddress (addr_t vm_addr)
{
    SBBreakpointLocation sb_bp_location;

    if (m_opaque_sp)
    {
        if (vm_addr != LLDB_INVALID_ADDRESS)
        {
            Address address;
            Target &target = m_opaque_sp->GetTarget();
            if (target.GetSectionLoadList().ResolveLoadAddress (vm_addr, address) == false)
            {
                address.SetSection (NULL);
                address.SetOffset (vm_addr);
            }
            sb_bp_location.SetLocation (m_opaque_sp->FindLocationByAddress (address));
        }
    }
    return sb_bp_location;
}

break_id_t
SBBreakpoint::FindLocationIDByAddress (addr_t vm_addr)
{
    break_id_t lldb_id = (break_id_t) 0;

    if (m_opaque_sp)
    {
        if (vm_addr != LLDB_INVALID_ADDRESS)
        {
            Address address;
            Target &target = m_opaque_sp->GetTarget();
            if (target.GetSectionLoadList().ResolveLoadAddress (vm_addr, address) == false)
            {
                address.SetSection (NULL);
                address.SetOffset (vm_addr);
            }
            lldb_id = m_opaque_sp->FindLocationIDByAddress (address);
        }
    }

    return lldb_id;
}

SBBreakpointLocation
SBBreakpoint::FindLocationByID (break_id_t bp_loc_id)
{
    SBBreakpointLocation sb_bp_location;

    if (m_opaque_sp)
        sb_bp_location.SetLocation (m_opaque_sp->FindLocationByID (bp_loc_id));

    return sb_bp_location;
}

SBBreakpointLocation
SBBreakpoint::GetLocationAtIndex (uint32_t index)
{
    SBBreakpointLocation sb_bp_location;

    if (m_opaque_sp)
        sb_bp_location.SetLocation (m_opaque_sp->GetLocationAtIndex (index));

    return sb_bp_location;
}

void
SBBreakpoint::SetEnabled (bool enable)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBBreakpoint::SetEnabled (this.sp=%p, enable='%s')", m_opaque_sp.get(), 
                     (enable ? "true" : "false"));

    if (m_opaque_sp)
        m_opaque_sp->SetEnabled (enable);
}

bool
SBBreakpoint::IsEnabled ()
{
    if (m_opaque_sp)
        return m_opaque_sp->IsEnabled();
    else
        return false;
}

void
SBBreakpoint::SetIgnoreCount (uint32_t count)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBBreakpoint::SetIgnoreCount (this.sp=%p, count='%d')", m_opaque_sp.get(), count);
        
    if (m_opaque_sp)
        m_opaque_sp->SetIgnoreCount (count);
}

void
SBBreakpoint::SetCondition (const char *condition)
{
    m_opaque_sp->SetCondition (condition);
}

const char *
SBBreakpoint::GetCondition ()
{
    return m_opaque_sp->GetConditionText ();
}

uint32_t
SBBreakpoint::GetHitCount () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBBreakpoint::GetHitCount");
        
    if (m_opaque_sp)
    {
        uint32_t hit_count = m_opaque_sp->GetHitCount();
        if (log)
            log->Printf ("SBBreakpoint::GetHitCount (this.sp=%p) => '%d'", m_opaque_sp.get(), hit_count);        
        return m_opaque_sp->GetHitCount();
    }
    else
    {
        if (log)
            log->Printf ("SBBreakpoint::GetHitCount (this.sp=%p) => '0'", m_opaque_sp.get());
        return 0;
    }
}

uint32_t
SBBreakpoint::GetIgnoreCount () const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetIgnoreCount();
    else
        return 0;
}

void
SBBreakpoint::SetThreadID (tid_t sb_thread_id)
{
    if (m_opaque_sp)
        m_opaque_sp->SetThreadID (sb_thread_id);
}

tid_t
SBBreakpoint::GetThreadID ()
{
    tid_t lldb_thread_id = LLDB_INVALID_THREAD_ID;
    if (m_opaque_sp)
        lldb_thread_id = m_opaque_sp->GetThreadID();

    return lldb_thread_id;
}

void
SBBreakpoint::SetThreadIndex (uint32_t index)
{
    if (m_opaque_sp)
        m_opaque_sp->GetOptions()->GetThreadSpec()->SetIndex (index);
}

uint32_t
SBBreakpoint::GetThreadIndex() const
{
    if (m_opaque_sp)
    {
        const ThreadSpec *thread_spec = m_opaque_sp->GetOptions()->GetThreadSpec();
        if (thread_spec == NULL)
            return 0;
        else
            return thread_spec->GetIndex();
    }
    return 0;
}
    

void
SBBreakpoint::SetThreadName (const char *thread_name)
{
    if (m_opaque_sp)
        m_opaque_sp->GetOptions()->GetThreadSpec()->SetName (thread_name);
}

const char *
SBBreakpoint::GetThreadName () const
{
    if (m_opaque_sp)
    {
        const ThreadSpec *thread_spec = m_opaque_sp->GetOptions()->GetThreadSpec();
        if (thread_spec == NULL)
            return NULL;
        else
            return thread_spec->GetName();
    }
    return NULL;
}

void
SBBreakpoint::SetQueueName (const char *queue_name)
{
    if (m_opaque_sp)
        m_opaque_sp->GetOptions()->GetThreadSpec()->SetQueueName (queue_name);
}

const char *
SBBreakpoint::GetQueueName () const
{
    if (m_opaque_sp)
    {
        const ThreadSpec *thread_spec = m_opaque_sp->GetOptions()->GetThreadSpec();
        if (thread_spec == NULL)
            return NULL;
        else
            return thread_spec->GetQueueName();
    }
    return NULL;
}

size_t
SBBreakpoint::GetNumResolvedLocations() const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetNumResolvedLocations();
    else
        return 0;
}

size_t
SBBreakpoint::GetNumLocations() const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetNumLocations();
    else
        return 0;
}

bool
SBBreakpoint::GetDescription (SBStream &s)
{
    if (m_opaque_sp)
    {
        s.Printf("SBBreakpoint: id = %i, ", m_opaque_sp->GetID());
        m_opaque_sp->GetResolverDescription (s.get());
        m_opaque_sp->GetFilterDescription (s.get());
        const size_t num_locations = m_opaque_sp->GetNumLocations ();
        s.Printf(", locations = %zu", num_locations);
        return true;
    }
    s.Printf ("No value");
    return false;
}

bool
SBBreakpoint::PrivateBreakpointHitCallback 
(
    void *baton, 
    StoppointCallbackContext *ctx, 
    lldb::user_id_t break_id, 
    lldb::user_id_t break_loc_id
)
{
    BreakpointSP bp_sp(ctx->exe_ctx.target->GetBreakpointList().FindBreakpointByID(break_id));
    if (baton && bp_sp)
    {
        CallbackData *data = (CallbackData *)baton;
        lldb_private::Breakpoint *bp = bp_sp.get();
        if (bp && data->callback)
        {
            if (ctx->exe_ctx.process)
            {
                SBProcess sb_process (ctx->exe_ctx.process->GetSP());
                SBThread sb_thread;
                SBBreakpointLocation sb_location;
                assert (bp_sp);
                sb_location.SetLocation (bp_sp->FindLocationByID (break_loc_id));
                if (ctx->exe_ctx.thread)
                    sb_thread.SetThread(ctx->exe_ctx.thread->GetSP());

                return data->callback (data->callback_baton, 
                                          sb_process, 
                                          sb_thread, 
                                          sb_location);
            }
        }
    }
    return true;    // Return true if we should stop at this breakpoint
}

void
SBBreakpoint::SetCallback (BreakpointHitCallback callback, void *baton)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    
    if (log)
        log->Printf ("SBBreakpoint::SetCallback (this.sp=%p, :", m_opaque_sp.get());

    if (m_opaque_sp.get())
    {
        BatonSP baton_sp(new SBBreakpointCallbackBaton (callback, baton));
        if (log)
        {
            // CAROLINE: FIXME!!
            //StreamString sstr;
            //baton_sp->GetDescription (sstr, lldb::eDescriptionLevelFull);
            //log->Printf ("%s", sstr.GetData());
        }
        m_opaque_sp->SetCallback (SBBreakpoint::PrivateBreakpointHitCallback, baton_sp, false);
    }
}


lldb_private::Breakpoint *
SBBreakpoint::operator->() const
{
    return m_opaque_sp.get();
}

lldb_private::Breakpoint *
SBBreakpoint::get() const
{
    return m_opaque_sp.get();
}

lldb::BreakpointSP &
SBBreakpoint::operator *()
{
    return m_opaque_sp;
}

const lldb::BreakpointSP &
SBBreakpoint::operator *() const
{
    return m_opaque_sp;
}

BreakpointEventType
SBBreakpoint::GetBreakpointEventTypeFromEvent (const SBEvent& event)
{
    if (event.IsValid())
        return Breakpoint::BreakpointEventData::GetBreakpointEventTypeFromEvent (event.GetSP());
    return eBreakpointEventTypeInvalidType;
}

SBBreakpoint
SBBreakpoint::GetBreakpointFromEvent (const lldb::SBEvent& event)
{
    SBBreakpoint sb_breakpoint;
    if (event.IsValid())
        sb_breakpoint.m_opaque_sp = Breakpoint::BreakpointEventData::GetBreakpointFromEvent (event.GetSP());
    return sb_breakpoint;
}

SBBreakpointLocation
SBBreakpoint::GetBreakpointLocationAtIndexFromEvent (const lldb::SBEvent& event, uint32_t loc_idx)
{
    SBBreakpointLocation sb_breakpoint_loc;
    if (event.IsValid())
        sb_breakpoint_loc.SetLocation (Breakpoint::BreakpointEventData::GetBreakpointLocationAtIndexFromEvent (event.GetSP(), loc_idx));
    return sb_breakpoint_loc;
}


