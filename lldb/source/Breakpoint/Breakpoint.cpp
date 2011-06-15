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
#include "lldb/Breakpoint/BreakpointLocationCollection.h"
#include "lldb/Breakpoint/BreakpointResolver.h"
#include "lldb/Breakpoint/BreakpointResolverFileLine.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadSpec.h"
#include "lldb/lldb-private-log.h"
#include "llvm/Support/Casting.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

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
Breakpoint::AddLocation (const Address &addr, bool *new_location)
{
    if (new_location)
        *new_location = false;
    BreakpointLocationSP bp_loc_sp (m_locations.FindByAddress(addr));
    if (!bp_loc_sp)
	{
		bp_loc_sp = m_locations.Create (*this, addr);
		if (bp_loc_sp)
		{
	    	bp_loc_sp->ResolveBreakpointSite();

		    if (new_location)
	    	    *new_location = true;
		}
	}
    return bp_loc_sp;
}

BreakpointLocationSP
Breakpoint::FindLocationByAddress (const Address &addr)
{
    return m_locations.FindByAddress(addr);
}

break_id_t
Breakpoint::FindLocationIDByAddress (const Address &addr)
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
Breakpoint::SetIgnoreCount (uint32_t n)
{
    m_options.SetIgnoreCount(n);
}

uint32_t
Breakpoint::GetIgnoreCount () const
{
    return m_options.GetIgnoreCount();
}

uint32_t
Breakpoint::GetHitCount () const
{
    return m_locations.GetHitCount();
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

void 
Breakpoint::SetCondition (const char *condition)
{
    m_options.SetCondition (condition);
}

ThreadPlan *
Breakpoint::GetThreadPlanToTestCondition (ExecutionContext &exe_ctx, lldb::BreakpointLocationSP loc_sp, Stream &error)
{
    return m_options.GetThreadPlanToTestCondition (exe_ctx, loc_sp, error);
}

const char *
Breakpoint::GetConditionText () const
{
    return m_options.GetConditionText();
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

        const size_t num_locs = m_locations.GetSize();

        for (size_t i = 0; i < module_list.GetSize(); i++)
        {
            bool seen = false;
            ModuleSP module_sp (module_list.GetModuleAtIndex (i));
            if (!m_filter_sp->ModulePasses (module_sp))
                continue;

            for (size_t loc_idx = 0; loc_idx < num_locs; loc_idx++)
            {
                BreakpointLocationSP break_loc = m_locations.GetByIndex(loc_idx);
                if (!break_loc->IsEnabled())
                    continue;
                const Section *section = break_loc->GetAddress().GetSection();
                if (section == NULL || section->GetModule() == module_sp.get())
                {
                    if (!seen)
                        seen = true;

                    if (!break_loc->ResolveBreakpointSite())
                    {
                        LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
                        if (log)
                            log->Printf ("Warning: could not set breakpoint site for breakpoint location %d of breakpoint %d.\n",
                                         break_loc->GetID(), GetID());
                    }
                }
            }

            if (!seen)
                new_modules.AppendIfNeeded (module_sp);

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

        for (size_t i = 0; i < module_list.GetSize(); i++)
        {
            ModuleSP module_sp (module_list.GetModuleAtIndex (i));
            if (m_filter_sp->ModulePasses (module_sp))
            {
                const size_t num_locs = m_locations.GetSize();
                for (size_t loc_idx = 0; loc_idx < num_locs; ++loc_idx)
                {
                    BreakpointLocationSP break_loc = m_locations.GetByIndex(loc_idx);
                    const Section *section = break_loc->GetAddress().GetSection();
                    if (section && section->GetModule() == module_sp.get())
                    {
                        // Remove this breakpoint since the shared library is 
                        // unloaded, but keep the breakpoint location around
                        // so we always get complete hit count and breakpoint
                        // lifetime info
                        break_loc->ClearBreakpointSite();
                    }
                }
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
    s->Printf("%i: ", GetID());
    GetResolverDescription (s);
    GetFilterDescription (s);

    const size_t num_locations = GetNumLocations ();
    const size_t num_resolved_locations = GetNumResolvedLocations ();

    switch (level)
    {
    case lldb::eDescriptionLevelBrief:
    case lldb::eDescriptionLevelFull:
        if (num_locations > 0)
        {
            s->Printf(", locations = %zu", num_locations);
            if (num_resolved_locations > 0)
                s->Printf(", resolved = %zu", num_resolved_locations);
        }
        else
        {
            s->Printf(", locations = 0 (pending)");
        }

        GetOptions()->GetDescription(s, level);
        
        if (level == lldb::eDescriptionLevelFull)
        {
            s->IndentLess();
            s->EOL();
        }
        break;

    case lldb::eDescriptionLevelVerbose:
        // Verbose mode does a debug dump of the breakpoint
        Dump (s);
        s->EOL ();
            //s->Indent();
        GetOptions()->GetDescription(s, level);
        break;

    default: 
        break;
    }

    // The brief description is just the location name (1.2 or whatever).  That's pointless to
    // show in the breakpoint's description, so suppress it.
    if (show_locations && level != lldb::eDescriptionLevelBrief)
    {
        s->IndentMore();
        for (size_t i = 0; i < num_locations; ++i)
        {
            BreakpointLocation *loc = GetLocationAtIndex(i).get();
            loc->GetDescription(s, level);
            s->EOL();
        }
        s->IndentLess();
    }
}

Breakpoint::BreakpointEventData::BreakpointEventData (BreakpointEventType sub_type, 
                                                      BreakpointSP &new_breakpoint_sp) :
    EventData (),
    m_breakpoint_event (sub_type),
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

BreakpointEventType
Breakpoint::BreakpointEventData::GetBreakpointEventType () const
{
    return m_breakpoint_event;
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

BreakpointEventType
Breakpoint::BreakpointEventData::GetBreakpointEventTypeFromEvent (const EventSP &event_sp)
{
    BreakpointEventData *data = GetEventDataFromEvent (event_sp);

    if (data == NULL)
        return eBreakpointEventTypeInvalidType;
    else
        return data->GetBreakpointEventType();
}

BreakpointSP
Breakpoint::BreakpointEventData::GetBreakpointFromEvent (const EventSP &event_sp)
{
    BreakpointSP bp_sp;

    BreakpointEventData *data = GetEventDataFromEvent (event_sp);
    if (data)
        bp_sp = data->GetBreakpoint();

    return bp_sp;
}

lldb::BreakpointLocationSP
Breakpoint::BreakpointEventData::GetBreakpointLocationAtIndexFromEvent (const lldb::EventSP &event_sp, uint32_t bp_loc_idx)
{
    lldb::BreakpointLocationSP bp_loc_sp;

    BreakpointEventData *data = GetEventDataFromEvent (event_sp);
    if (data)
    {
        Breakpoint *bp = data->GetBreakpoint().get();
        if (bp)
            bp_loc_sp = bp->GetLocationAtIndex(bp_loc_idx);
    }

    return bp_loc_sp;
}


void
Breakpoint::GetResolverDescription (Stream *s)
{
    if (m_resolver_sp)
        m_resolver_sp->GetDescription (s);
}


bool
Breakpoint::GetMatchingFileLine (const ConstString &filename, uint32_t line_number, BreakpointLocationCollection &loc_coll)
{
    // TODO: To be correct, this method needs to fill the breakpoint location collection
    //       with the location IDs which match the filename and line_number.
    //

    if (m_resolver_sp)
    {
        BreakpointResolverFileLine *resolverFileLine = dyn_cast<BreakpointResolverFileLine>(m_resolver_sp.get());
        if (resolverFileLine &&
            resolverFileLine->m_file_spec.GetFilename() == filename &&
            resolverFileLine->m_line_number == line_number)
        {
            return true;
        }
    }
    return false;
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
