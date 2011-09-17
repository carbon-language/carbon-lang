//===-- BreakpointLocation.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/BreakpointID.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Core/StreamString.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;

BreakpointLocation::BreakpointLocation
(
    break_id_t loc_id,
    Breakpoint &owner,
    const Address &addr,
    lldb::tid_t tid,
    bool hardware
) :
    StoppointLocation (loc_id, addr.GetOpcodeLoadAddress(&owner.GetTarget()), hardware),
    m_address (addr),
    m_owner (owner),
    m_options_ap (),
    m_bp_site_sp ()
{
    SetThreadID (tid);
}

BreakpointLocation::~BreakpointLocation()
{
    ClearBreakpointSite();
}

lldb::addr_t
BreakpointLocation::GetLoadAddress () const
{
    return m_address.GetOpcodeLoadAddress (&m_owner.GetTarget());
}

Address &
BreakpointLocation::GetAddress ()
{
    return m_address;
}

Breakpoint &
BreakpointLocation::GetBreakpoint ()
{
    return m_owner;
}

bool
BreakpointLocation::IsEnabled ()
{
    if (!m_owner.IsEnabled())
        return false;
    else if (m_options_ap.get() != NULL)
        return m_options_ap->IsEnabled();
    else
        return true;
}

void
BreakpointLocation::SetEnabled (bool enabled)
{
    GetLocationOptions()->SetEnabled(enabled);
    if (enabled)
    {
        ResolveBreakpointSite();
    }
    else
    {
        ClearBreakpointSite();
    }
}

void
BreakpointLocation::SetThreadID (lldb::tid_t thread_id)
{
    if (thread_id != LLDB_INVALID_THREAD_ID)
        GetLocationOptions()->SetThreadID(thread_id);
    else
    {
        // If we're resetting this to an invalid thread id, then
        // don't make an options pointer just to do that.
        if (m_options_ap.get() != NULL)
            m_options_ap->SetThreadID (thread_id);
    }
}

bool
BreakpointLocation::InvokeCallback (StoppointCallbackContext *context)
{
    if (m_options_ap.get() != NULL && m_options_ap->HasCallback())
        return m_options_ap->InvokeCallback (context, m_owner.GetID(), GetID());
    else    
        return m_owner.InvokeCallback (context, GetID());
}

void
BreakpointLocation::SetCallback (BreakpointHitCallback callback, void *baton,
                 bool is_synchronous)
{
    // The default "Baton" class will keep a copy of "baton" and won't free
    // or delete it when it goes goes out of scope.
    GetLocationOptions()->SetCallback(callback, BatonSP (new Baton(baton)), is_synchronous);
}

void
BreakpointLocation::SetCallback (BreakpointHitCallback callback, const BatonSP &baton_sp,
                 bool is_synchronous)
{
    GetLocationOptions()->SetCallback (callback, baton_sp, is_synchronous);
}


void
BreakpointLocation::ClearCallback ()
{
    GetLocationOptions()->ClearCallback();
}

void 
BreakpointLocation::SetCondition (const char *condition)
{
    GetLocationOptions()->SetCondition (condition);
}

ThreadPlan *
BreakpointLocation::GetThreadPlanToTestCondition (ExecutionContext &exe_ctx, Stream &error)
{
    lldb::BreakpointLocationSP this_sp(this);
    if (m_options_ap.get())
        return m_options_ap->GetThreadPlanToTestCondition (exe_ctx, this_sp, error);
    else
        return m_owner.GetThreadPlanToTestCondition (exe_ctx, this_sp, error);
}

const char *
BreakpointLocation::GetConditionText () const
{
    return GetOptionsNoCreate()->GetConditionText();
}

uint32_t
BreakpointLocation::GetIgnoreCount ()
{
    return GetOptionsNoCreate()->GetIgnoreCount();
}

void
BreakpointLocation::SetIgnoreCount (uint32_t n)
{
    GetLocationOptions()->SetIgnoreCount(n);
}

const BreakpointOptions *
BreakpointLocation::GetOptionsNoCreate () const
{
    if (m_options_ap.get() != NULL)
        return m_options_ap.get();
    else
        return m_owner.GetOptions ();
}

BreakpointOptions *
BreakpointLocation::GetLocationOptions ()
{
    // If we make the copy we don't copy the callbacks because that is potentially 
    // expensive and we don't want to do that for the simple case where someone is
    // just disabling the location.
    if (m_options_ap.get() == NULL)
        m_options_ap.reset(BreakpointOptions::CopyOptionsNoCallback(*m_owner.GetOptions ()));
    
    return m_options_ap.get();
}

bool
BreakpointLocation::ValidForThisThread (Thread *thread)
{
    return thread->MatchesSpec(GetOptionsNoCreate()->GetThreadSpecNoCreate());
}

// RETURNS - true if we should stop at this breakpoint, false if we
// should continue.  Note, we don't check the thread spec for the breakpoint
// here, since if the breakpoint is not for this thread, then the event won't
// even get reported, so the check is redundant.

bool
BreakpointLocation::ShouldStop (StoppointCallbackContext *context)
{
    bool should_stop = true;
    LogSP log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);

    m_hit_count++;

    if (!IsEnabled())
        return false;

    if (m_hit_count <= GetIgnoreCount())
        return false;

    // We only run synchronous callbacks in ShouldStop:
    context->is_synchronous = true;
    should_stop = InvokeCallback (context);
    
    if (log)
    {
        StreamString s;
        GetDescription (&s, lldb::eDescriptionLevelVerbose);
        log->Printf ("Hit breakpoint location: %s, %s.\n", s.GetData(), should_stop ? "stopping" : "continuing");
    }
    
    return should_stop;
}

bool
BreakpointLocation::IsResolved () const
{
    return m_bp_site_sp.get() != NULL;
}

lldb::BreakpointSiteSP
BreakpointLocation::GetBreakpointSite() const
{
    return m_bp_site_sp;
}

bool
BreakpointLocation::ResolveBreakpointSite ()
{
    if (m_bp_site_sp)
        return true;

    Process *process = m_owner.GetTarget().GetProcessSP().get();
    if (process == NULL)
        return false;

    if (m_owner.GetTarget().GetSectionLoadList().IsEmpty())
        return false;

    BreakpointLocationSP this_sp(this);

    lldb::break_id_t new_id = process->CreateBreakpointSite (this_sp, false);

    if (new_id == LLDB_INVALID_BREAK_ID)
    {
        LogSP log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);
        if (log)
            log->Warning ("Tried to add breakpoint site at 0x%llx but it was already present.\n",
                          m_address.GetOpcodeLoadAddress (&m_owner.GetTarget()));
        return false;
    }

    return true;
}

bool
BreakpointLocation::SetBreakpointSite (BreakpointSiteSP& bp_site_sp)
{
    m_bp_site_sp = bp_site_sp;
    return true;
}

bool
BreakpointLocation::ClearBreakpointSite ()
{
    if (m_bp_site_sp.get())
    {
        m_owner.GetTarget().GetProcessSP()->RemoveOwnerFromBreakpointSite (GetBreakpoint().GetID(), 
                                                                           GetID(), m_bp_site_sp);
        m_bp_site_sp.reset();
        return true;
    }
    return false;
}

void
BreakpointLocation::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    SymbolContext sc;
    s->Indent();
    BreakpointID::GetCanonicalReference(s, m_owner.GetID(), GetID());

    if (level == lldb::eDescriptionLevelBrief)
        return;

    s->PutCString(": ");

    if (level == lldb::eDescriptionLevelVerbose)
        s->IndentMore();

    if (m_address.IsSectionOffset())
    {
        m_address.CalculateSymbolContext(&sc);

        if (level == lldb::eDescriptionLevelFull)
        {
            s->PutCString("where = ");
            sc.DumpStopContext (s, m_owner.GetTarget().GetProcessSP().get(), m_address, false, true, false);
        }
        else
        {
            if (sc.module_sp)
            {
                s->EOL();
                s->Indent("module = ");
                sc.module_sp->GetFileSpec().Dump (s);
            }

            if (sc.comp_unit != NULL)
            {
                s->EOL();
                s->Indent("compile unit = ");
                static_cast<FileSpec*>(sc.comp_unit)->GetFilename().Dump (s);

                if (sc.function != NULL)
                {
                    s->EOL();
                    s->Indent("function = ");
                    s->PutCString (sc.function->GetMangled().GetName().AsCString("<unknown>"));
                }

                if (sc.line_entry.line > 0)
                {
                    s->EOL();
                    s->Indent("location = ");
                    sc.line_entry.DumpStopContext (s, true);
                }

            }
            else
            {
                // If we don't have a comp unit, see if we have a symbol we can print.
                if (sc.symbol)
                {
                    s->EOL();
                    s->Indent("symbol = ");
                    s->PutCString(sc.symbol->GetMangled().GetName().AsCString("<unknown>"));
                }
            }
        }
    }

    if (level == lldb::eDescriptionLevelVerbose)
    {
        s->EOL();
        s->Indent();
    }
    s->Printf ("%saddress = ", (level == lldb::eDescriptionLevelFull && m_address.IsSectionOffset()) ? ", " : "");
    ExecutionContextScope *exe_scope = NULL;
    Target *target = &m_owner.GetTarget();
    if (target)
        exe_scope = target->GetProcessSP().get();
    if (exe_scope == NULL)
        exe_scope = target;

    m_address.Dump(s, exe_scope, Address::DumpStyleLoadAddress, Address::DumpStyleModuleWithFileAddress);

    if (level == lldb::eDescriptionLevelVerbose)
    {
        s->EOL();
        s->Indent();
        s->Printf("resolved = %s\n", IsResolved() ? "true" : "false");

        s->Indent();
        s->Printf ("hit count = %-4u\n", GetHitCount());

        if (m_options_ap.get())
        {
            s->Indent();
            m_options_ap->GetDescription (s, level);
            s->EOL();
        }
        s->IndentLess();
    }
    else
    {
        s->Printf(", %sresolved, hit count = %u ",
                  (IsResolved() ? "" : "un"),
                  GetHitCount());
        if (m_options_ap.get())
        {
            m_options_ap->GetDescription (s, level);
        }
    }
}

void
BreakpointLocation::Dump(Stream *s) const
{
    if (s == NULL)
        return;

    s->Printf("BreakpointLocation %u: tid = %4.4x  load addr = 0x%8.8llx  state = %s  type = %s breakpoint  "
              "hw_index = %i  hit_count = %-4u  ignore_count = %-4u",
            GetID(),
            GetOptionsNoCreate()->GetThreadSpecNoCreate()->GetTID(),
            (uint64_t) m_address.GetOpcodeLoadAddress (&m_owner.GetTarget()),
            (m_options_ap.get() ? m_options_ap->IsEnabled() : m_owner.IsEnabled()) ? "enabled " : "disabled",
            IsHardware() ? "hardware" : "software",
            GetHardwareIndex(),
            GetHitCount(),
            m_options_ap.get() ? m_options_ap->GetIgnoreCount() : m_owner.GetIgnoreCount());
}
