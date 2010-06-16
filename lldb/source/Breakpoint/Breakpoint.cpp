//===-- Breakpoint.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/Core/Address.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/BreakpointResolver.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadSpec.h"
#include "lldb/lldb-private-log.h"

using namespace lldb;
using namespace lldb_private;

const ConstString &
Breakpoint::GetEventIdentifier ()
{
    static ConstString g_identifier("event-identifier.breakpoint.changed");
    return g_identifier;
}

//----------------------------------------------------------------------
// Breakpoint constructor
//----------------------------------------------------------------------
Breakpoint::Breakpoint(Target &target, SearchFilterSP &filter_sp, BreakpointResolverSP &resolver_sp) :
    m_target (target),
    m_filter_sp (filter_sp),
    m_resolver_sp (resolver_sp),
    m_options (),
    m_locations ()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Breakpoint::~Breakpoint()
{
}

bool
Breakpoint::IsInternal () const
{
    return LLDB_BREAK_ID_IS_INTERNAL(m_bid);
}



Target&
Breakpoint::GetTarget ()
{
    return m_target;
}

const Target&
Breakpoint::GetTarget () const
{
    return m_target;
}

BreakpointLocationSP
Breakpoint::AddLocation (Address &addr, bool *new_location)
{
    BreakpointLocationSP bp_loc_sp (m_locations.FindByAddress(addr));
    if (bp_loc_sp)
    {
        if (new_location)
            *new_location = false;
        return bp_loc_sp;
    }

    bp_loc_sp.reset (new BreakpointLocation (m_locations.GetNextID(), *this, addr));
    m_locations.Add (bp_loc_sp);
    bp_loc_sp->ResolveBreakpointSite();

    if (new_location)
        *new_location = true;
    return bp_loc_sp;
}

BreakpointLocationSP
Breakpoint::FindLocationByAddress (Address &addr)
{
    return m_locations.FindByAddress(addr);
}

break_id_t
Breakpoint::FindLocationIDByAddress (Address &addr)
{
    return m_locations.FindIDByAddress(addr);
}

BreakpointLocationSP
Breakpoint::FindLocationByID (break_id_t bp_loc_id)
{
    return m_locations.FindByID(bp_loc_id);
}

BreakpointLocationSP
Breakpoint::GetLocationAtIndex (uint32_t index)
{
    return m_locations.GetByIndex(index);
}

BreakpointLocationSP
Breakpoint::GetLocationSP (BreakpointLocation *bp_loc_ptr)
{
    assert (bp_loc_ptr->GetBreakpoint().GetID() == GetID());
    return m_locations.FindByID(bp_loc_ptr->GetID());
}


// For each of the overall options we need to decide how they propagate to
// the location options.  This will determine the precedence of options on
// the breakpoint vrs. its locations.

// Disable at the breakpoint level should override the location settings.
// That way you can conveniently turn off a whole breakpoint without messing
// up the individual settings.

void
Breakpoint::SetEnabled (bool enable)
{
    m_options.SetEnabled(enable);
    if (enable)
        m_locations.ResolveAllBreakpointSites();
    else
        m_locations.ClearAllBreakpointSites();
}

bool
Breakpoint::IsEnabled ()
{
    return m_options.IsEnabled();
}

void
Breakpoint::SetIgnoreCount (int32_t n)
{
    m_options.SetIgnoreCount(n);
}

int32_t
Breakpoint::GetIgnoreCount () const
{
    return m_options.GetIgnoreCount();
}

void
Breakpoint::SetThreadID (lldb::tid_t thread_id)
{
    m_options.GetThreadSpec()->SetTID(thread_id);
}

lldb::tid_t
Breakpoint::GetThreadID ()
{
    if (m_options.GetThreadSpec() == NULL)
        return LLDB_INVALID_THREAD_ID;
    else
        return m_options.GetThreadSpec()->GetTID();
}

// This function is used when "baton" doesn't need to be freed
void
Breakpoint::SetCallback (BreakpointHitCallback callback, void *baton, bool is_synchronous)
{
    // The default "Baton" class will keep a copy of "baton" and won't free
    // or delete it when it goes goes out of scope.
    m_options.SetCallback(callback, BatonSP (new Baton(baton)), is_synchronous);
}

// This function is used when a baton needs to be freed and therefore is 
// contained in a "Baton" subclass.
void
Breakpoint::SetCallback (BreakpointHitCallback callback, const BatonSP &callback_baton_sp, bool is_synchronous)
{
    m_options.SetCallback(callback, callback_baton_sp, is_synchronous);
}

void
Breakpoint::ClearCallback ()
{
    m_options.ClearCallback ();
}

bool
Breakpoint::InvokeCallback (StoppointCallbackContext *context, break_id_t bp_loc_id)
{
    return m_options.InvokeCallback (context, GetID(), bp_loc_id);
}

BreakpointOptions *
Breakpoint::GetOptions ()
{
    return &m_options;
}

void
Breakpoint::ResolveBreakpoint ()
{
    if (m_resolver_sp)
        m_resolver_sp->ResolveBreakpoint(*m_filter_sp);
}

void
Breakpoint::ResolveBreakpointInModules (ModuleList &module_list)
{
    if (m_resolver_sp)
        m_resolver_sp->ResolveBreakpointInModules(*m_filter_sp, module_list);
}

void
Breakpoint::ClearAllBreakpointSites ()
{
    m_locations.ClearAllBreakpointSites();
}

//----------------------------------------------------------------------
// ModulesChanged: Pass in a list of new modules, and
//----------------------------------------------------------------------

void
Breakpoint::ModulesChanged (ModuleList &module_list, bool load)
{
    if (load)
    {
        // The logic for handling new modules is:
        // 1) If the filter rejects this module, then skip it.
        // 2) Run through the current location list and if there are any locations
        //    for that module, we mark the module as "seen" and we don't try to re-resolve
        //    breakpoint locations for that module.
        //    However, we do add breakpoint sites to these locations if needed.
        // 3) If we don't see this module in our breakpoint location list, call ResolveInModules.

        ModuleList new_modules;  // We'll stuff the "unseen" modules in this list, and then resolve
                                   // them after the locations pass.  Have to do it this way because
                                   // resolving breakpoints will add new locations potentially.

        for (int i = 0; i < module_list.GetSize(); i++)
        {
            bool seen = false;
            ModuleSP module_sp (module_list.GetModuleAtIndex (i));
            Module *module = module_sp.get();
            if (!m_filter_sp->ModulePasses (module_sp))
                continue;

            for (int i = 0; i < m_locations.GetSize(); i++)
            {
                BreakpointLocationSP break_loc = m_locations.GetByIndex(i);
                const Section *section = break_loc->GetAddress().GetSection();
                if (section == NULL || section->GetModule() == module)
                {
                    if (!seen)
                        seen = true;

                    if (!break_loc->ResolveBreakpointSite())
                    {
                        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);
                        if (log)
                            log->Printf ("Warning: could not set breakpoint site for breakpoint location %d of breakpoint %d.\n",
                                         break_loc->GetID(), GetID());
                    }
                }
            }

            if (!seen)
                new_modules.AppendInNeeded (module_sp);

        }
        if (new_modules.GetSize() > 0)
        {
            ResolveBreakpointInModules(new_modules);
        }
    }
    else
    {
        // Go through the currently set locations and if any have breakpoints in
        // the module list, then remove their breakpoint sites.
        // FIXME: Think about this...  Maybe it's better to delete the locations?
        // Are we sure that on load-unload-reload the module pointer will remain
        // the same?  Or do we need to do an equality on modules that is an
        // "equivalence"???

        for (int i = 0; i < module_list.GetSize(); i++)
        {
            ModuleSP module_sp (module_list.GetModuleAtIndex (i));
            if (!m_filter_sp->ModulePasses (module_sp))
                continue;

            for (int i = 0; i < m_locations.GetSize(); i++)
            {
                BreakpointLocationSP break_loc = m_locations.GetByIndex(i);
                const Section *section = break_loc->GetAddress().GetSection();
                if (section)
                {
                    if (section->GetModule() == module_sp.get())
                        break_loc->ClearBreakpointSite();
                }
//                else
//                {
//                    Address temp_addr;
//                    if (module->ResolveLoadAddress(break_loc->GetLoadAddress(), m_target->GetProcess(), temp_addr))
//                        break_loc->ClearBreakpointSite();
//                }
            }
        }
    }
}

void
Breakpoint::Dump (Stream *)
{
}

size_t
Breakpoint::GetNumResolvedLocations() const
{
    // Return the number of breakpoints that are actually resolved and set
    // down in the inferior process.
    return m_locations.GetNumResolvedLocations();
}

size_t
Breakpoint::GetNumLocations() const
{
    return m_locations.GetSize();
}

void
Breakpoint::GetDescription (Stream *s, lldb::DescriptionLevel level, bool show_locations)
{
    assert (s != NULL);
    StreamString filter_strm;


    s->Printf("%i ", GetID());
    GetResolverDescription (s);
    GetFilterDescription (&filter_strm);
    if (filter_strm.GetString().compare ("No Filter") != 0)
    {
        s->Printf (", ");
        GetFilterDescription (s);
    }

    const uint32_t num_locations = GetNumLocations ();
    const uint32_t num_resolved_locations = GetNumResolvedLocations ();

    switch (level)
    {
    case lldb::eDescriptionLevelBrief:
    case lldb::eDescriptionLevelFull:
        if (num_locations > 0)
        {
            s->Printf(" with %u location%s", num_locations, num_locations > 1 ? "s" : "");
            if (num_resolved_locations > 0)
                s->Printf(" (%u resolved)", num_resolved_locations);
            s->PutChar(';');
        }
        else
        {
            s->Printf(" with 0 locations (Pending Breakpoint).");
        }

        if (level == lldb::eDescriptionLevelFull)
        {
            Baton *baton = GetOptions()->GetBaton();
            if (baton)
            {
                s->EOL ();
                s->Indent();
                baton->GetDescription(s, level);
            }
        }
        break;

    case lldb::eDescriptionLevelVerbose:
        // Verbose mode does a debug dump of the breakpoint
        Dump (s);
        Baton *baton = GetOptions()->GetBaton();
        if (baton)
        {
            s->EOL ();
            s->Indent();
            baton->GetDescription(s, level);
        }
        break;
    }

    if (show_locations)
    {
        s->EOL();
        s->IndentMore();
        for (int i = 0; i < GetNumLocations(); ++i)
        {
            BreakpointLocation *loc = GetLocationAtIndex(i).get();
            loc->GetDescription(s, level);
            s->EOL();
        }
        s->IndentLess();

    }
}

Breakpoint::BreakpointEventData::BreakpointEventData (Breakpoint::BreakpointEventData::EventSubType sub_type, BreakpointSP &new_breakpoint_sp) :
    EventData (),
    m_sub_type (sub_type),
    m_new_breakpoint_sp (new_breakpoint_sp)
{
}

Breakpoint::BreakpointEventData::~BreakpointEventData ()
{
}

const ConstString &
Breakpoint::BreakpointEventData::GetFlavorString ()
{
    static ConstString g_flavor ("Breakpoint::BreakpointEventData");
    return g_flavor;
}

const ConstString &
Breakpoint::BreakpointEventData::GetFlavor () const
{
    return BreakpointEventData::GetFlavorString ();
}


BreakpointSP &
Breakpoint::BreakpointEventData::GetBreakpoint ()
{
    return m_new_breakpoint_sp;
}

Breakpoint::BreakpointEventData::EventSubType
Breakpoint::BreakpointEventData::GetSubType () const
{
    return m_sub_type;
}

void
Breakpoint::BreakpointEventData::Dump (Stream *s) const
{
}

Breakpoint::BreakpointEventData *
Breakpoint::BreakpointEventData::GetEventDataFromEvent (const EventSP &event_sp)
{
    if (event_sp)
    {
        EventData *event_data = event_sp->GetData();
        if (event_data && event_data->GetFlavor() == BreakpointEventData::GetFlavorString())
            return static_cast <BreakpointEventData *> (event_sp->GetData());
    }
    return NULL;
}

Breakpoint::BreakpointEventData::EventSubType
Breakpoint::BreakpointEventData::GetSubTypeFromEvent (const EventSP &event_sp)
{
    BreakpointEventData *data = GetEventDataFromEvent (event_sp);

    if (data == NULL)
        return eBreakpointInvalidType;
    else
        return data->GetSubType();
}

BreakpointSP
Breakpoint::BreakpointEventData::GetBreakpointFromEvent (const EventSP &event_sp)
{
    BreakpointEventData *data = GetEventDataFromEvent (event_sp);

    if (data == NULL)
    {
        BreakpointSP ret_val;
        return ret_val;
    }
    else
        return data->GetBreakpoint();
}


void
Breakpoint::GetResolverDescription (Stream *s)
{
    if (m_resolver_sp)
        m_resolver_sp->GetDescription (s);
}

void
Breakpoint::GetFilterDescription (Stream *s)
{
    m_filter_sp->GetDescription (s);
}

const BreakpointSP
Breakpoint::GetSP ()
{
    return m_target.GetBreakpointList().FindBreakpointByID (GetID());
}
