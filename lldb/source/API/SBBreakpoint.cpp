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
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBThread.h"

#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Address.h"
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
    m_break_sp ()
{
}

SBBreakpoint::SBBreakpoint (const SBBreakpoint& rhs) :
    m_break_sp (rhs.m_break_sp)
{
}


SBBreakpoint::SBBreakpoint (const lldb::BreakpointSP &bp_sp) :
    m_break_sp (bp_sp)
{
}

SBBreakpoint::~SBBreakpoint()
{
}

const SBBreakpoint &
SBBreakpoint::operator = (const SBBreakpoint& rhs)
{
    if (this != &rhs)
    {
        m_break_sp = rhs.m_break_sp;
    }
    return *this;
}

break_id_t
SBBreakpoint::GetID () const
{
    if (m_break_sp)
        return m_break_sp->GetID();
    return LLDB_INVALID_BREAK_ID;
}


bool
SBBreakpoint::IsValid() const
{
    return m_break_sp;
}

void
SBBreakpoint::Dump (FILE *f)
{
    if (m_break_sp)
    {
        if (f == NULL)
            f = SBDebugger::GetOutputFileHandle();
        if (f == NULL)
            return;
        lldb_private::StreamFile str (f);
        m_break_sp->Dump (&str);
    }
}

void
SBBreakpoint::ClearAllBreakpointSites ()
{
    if (m_break_sp)
        m_break_sp->ClearAllBreakpointSites ();
}

SBBreakpointLocation
SBBreakpoint::FindLocationByAddress (addr_t vm_addr)
{
    SBBreakpointLocation sb_bp_location;

    if (m_break_sp)
    {
        if (vm_addr != LLDB_INVALID_ADDRESS)
        {
            Address address;
            Process *sb_process = m_break_sp->GetTarget().GetProcessSP().get();
            if (sb_process == NULL || sb_process->ResolveLoadAddress (vm_addr, address) == false)
            {
                address.SetSection (NULL);
                address.SetOffset (vm_addr);
            }
            sb_bp_location.SetLocation (m_break_sp->FindLocationByAddress (address));
        }
    }
    return sb_bp_location;
}

break_id_t
SBBreakpoint::FindLocationIDByAddress (addr_t vm_addr)
{
    break_id_t lldb_id = (break_id_t) 0;

    if (m_break_sp)
    {
        if (vm_addr != LLDB_INVALID_ADDRESS)
        {
            Address address;
            Process *sb_process = m_break_sp->GetTarget().GetProcessSP().get();
            if (sb_process == NULL || sb_process->ResolveLoadAddress (vm_addr, address) == false)
            {
                address.SetSection (NULL);
                address.SetOffset (vm_addr);
            }
            lldb_id = m_break_sp->FindLocationIDByAddress (address);
        }
    }

    return lldb_id;
}

SBBreakpointLocation
SBBreakpoint::FindLocationByID (break_id_t bp_loc_id)
{
    SBBreakpointLocation sb_bp_location;

    if (m_break_sp)
        sb_bp_location.SetLocation (m_break_sp->FindLocationByID (bp_loc_id));

    return sb_bp_location;
}

SBBreakpointLocation
SBBreakpoint::GetLocationAtIndex (uint32_t index)
{
    SBBreakpointLocation sb_bp_location;

    if (m_break_sp)
        sb_bp_location.SetLocation (m_break_sp->GetLocationAtIndex (index));

    return sb_bp_location;
}

void
SBBreakpoint::ListLocations (FILE* f, const char *description_level)
{
   if (f == NULL)
       f = SBDebugger::GetOutputFileHandle();

   if (f == NULL)
       return;

    if (m_break_sp)
    {
        DescriptionLevel level;
        if (strcmp (description_level, "brief") == 0)
            level = eDescriptionLevelBrief;
        else if (strcmp (description_level, "full") == 0)
            level = eDescriptionLevelFull;
        else if (strcmp (description_level, "verbose") == 0)
            level = eDescriptionLevelVerbose;
        else
            level = eDescriptionLevelBrief;

        StreamFile str (f);

        str.IndentMore();
        int num_locs = m_break_sp->GetNumLocations();
        for (int i = 0; i < num_locs; ++i)
        {
            BreakpointLocation *loc = m_break_sp->GetLocationAtIndex (i).get();
            loc->GetDescription (&str, level);
            str.EOL();
        }
    }
}

void
SBBreakpoint::SetEnabled (bool enable)
{
    if (m_break_sp)
        m_break_sp->SetEnabled (enable);
}

bool
SBBreakpoint::IsEnabled ()
{
    if (m_break_sp)
        return m_break_sp->IsEnabled();
    else
        return false;
}

void
SBBreakpoint::SetIgnoreCount (int32_t count)
{
    if (m_break_sp)
        m_break_sp->SetIgnoreCount (count);
}

int32_t
SBBreakpoint::GetIgnoreCount () const
{
    if (m_break_sp)
        return m_break_sp->GetIgnoreCount();
    else
        return 0;
}

void
SBBreakpoint::SetThreadID (tid_t sb_thread_id)
{
    if (m_break_sp)
        m_break_sp->SetThreadID (sb_thread_id);
}

tid_t
SBBreakpoint::GetThreadID ()
{
    tid_t lldb_thread_id = LLDB_INVALID_THREAD_ID;
    if (m_break_sp)
        lldb_thread_id = m_break_sp->GetThreadID();

    return lldb_thread_id;
}

void
SBBreakpoint::SetThreadIndex (uint32_t index)
{
    if (m_break_sp)
        m_break_sp->GetOptions()->GetThreadSpec()->SetIndex (index);
}

uint32_t
SBBreakpoint::GetThreadIndex() const
{
    if (m_break_sp)
    {
        const ThreadSpec *thread_spec = m_break_sp->GetOptions()->GetThreadSpec();
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
    if (m_break_sp)
        m_break_sp->GetOptions()->GetThreadSpec()->SetName (thread_name);
}

const char *
SBBreakpoint::GetThreadName () const
{
    if (m_break_sp)
    {
        const ThreadSpec *thread_spec = m_break_sp->GetOptions()->GetThreadSpec();
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
    if (m_break_sp)
        m_break_sp->GetOptions()->GetThreadSpec()->SetQueueName (queue_name);
}

const char *
SBBreakpoint::GetQueueName () const
{
    if (m_break_sp)
    {
        const ThreadSpec *thread_spec = m_break_sp->GetOptions()->GetThreadSpec();
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
    if (m_break_sp)
        return m_break_sp->GetNumResolvedLocations();
    else
        return 0;
}

size_t
SBBreakpoint::GetNumLocations() const
{
    if (m_break_sp)
        return m_break_sp->GetNumLocations();
    else
        return 0;
}

void
SBBreakpoint::GetDescription (FILE *f, const char *description_level, bool describe_locations)
{
    if (f == NULL)
        return;

    if (m_break_sp)
    {
        DescriptionLevel level;
        if (strcmp (description_level, "brief") == 0)
            level = eDescriptionLevelBrief;
        else if (strcmp (description_level, "full") == 0)
            level = eDescriptionLevelFull;
        else if (strcmp (description_level, "verbose") == 0)
            level = eDescriptionLevelVerbose;
        else
            level = eDescriptionLevelBrief;

        StreamFile str (f);

        m_break_sp->GetDescription (&str, level);
        str.EOL();
        if (describe_locations)
        {
          //str.IndentMore();
          // int num_locs = m_break_sp->GetNumLocations();
          //  for (int i = 0; i < num_locs; ++i)
          //  {
          //      BreakpointLocation *loc = m_break_sp->FindLocationByIndex (i);
          //      loc->GetDescription (&str, level);
          //      str.EOL();
          //  }
            ListLocations (f, description_level);
        }
    }
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
    BreakpointSP bp_sp(ctx->context.target->GetBreakpointList().FindBreakpointByID(break_id));
    if (baton && bp_sp)
    {
        CallbackData *data = (CallbackData *)baton;
        lldb_private::Breakpoint *bp = bp_sp.get();
        if (bp && data->callback)
        {
            if (ctx->context.process)
            {
                SBProcess sb_process (ctx->context.process->GetSP());
                SBThread sb_thread;
                SBBreakpointLocation sb_location;
                assert (bp_sp);
                sb_location.SetLocation (bp_sp->FindLocationByID (break_loc_id));
                if (ctx->context.thread)
                    sb_thread.SetThread(ctx->context.thread->GetSP());

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
    if (m_break_sp.get())
    {
        BatonSP baton_sp(new SBBreakpointCallbackBaton (callback, baton));
        m_break_sp->SetCallback (SBBreakpoint::PrivateBreakpointHitCallback, baton_sp, false);
    }
}


lldb_private::Breakpoint *
SBBreakpoint::operator->() const
{
    return m_break_sp.get();
}

lldb_private::Breakpoint *
SBBreakpoint::get() const
{
    return m_break_sp.get();
}

lldb::BreakpointSP &
SBBreakpoint::operator *()
{
    return m_break_sp;
}

const lldb::BreakpointSP &
SBBreakpoint::operator *() const
{
    return m_break_sp;
}

