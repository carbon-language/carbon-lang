//===-- BreakpointLocationList.cpp ------------------------------*- C++ -*-===//
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
#include "lldb/Breakpoint/BreakpointLocationList.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

BreakpointLocationList::BreakpointLocationList() :
    m_locations(),
    m_address_to_location (),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_next_id (0)
{
}

BreakpointLocationList::~BreakpointLocationList()
{
}

lldb::break_id_t
BreakpointLocationList::Add (BreakpointLocationSP &bp_loc_sp)
{
    if (bp_loc_sp)
    {
        Mutex::Locker locker (m_mutex);
        m_locations.push_back (bp_loc_sp);
        m_address_to_location[bp_loc_sp->GetAddress()] = bp_loc_sp;
        return bp_loc_sp->GetID();
    }
    return LLDB_INVALID_BREAK_ID;
}

bool
BreakpointLocationList::ShouldStop (StoppointCallbackContext *context, lldb::break_id_t break_id)
{
    BreakpointLocationSP bp = FindByID (break_id);
    if (bp)
    {
        // Let the BreakpointLocation decide if it should stop here (could not have
        // reached it's target hit count yet, or it could have a callback
        // that decided it shouldn't stop (shared library loads/unloads).
        return bp->ShouldStop (context);
    }
    // We should stop here since this BreakpointLocation isn't valid anymore or it
    // doesn't exist.
    return true;
}

lldb::break_id_t
BreakpointLocationList::FindIDByAddress (Address &addr)
{
    BreakpointLocationSP bp_loc_sp = FindByAddress (addr);
    if (bp_loc_sp)
    {
        return bp_loc_sp->GetID();
    }
    return LLDB_INVALID_BREAK_ID;
}

bool
BreakpointLocationList::Remove (lldb::break_id_t break_id)
{
    Mutex::Locker locker (m_mutex);
    collection::iterator pos = GetIDIterator(break_id);    // Predicate
    if (pos != m_locations.end())
    {
        m_address_to_location.erase ((*pos)->GetAddress());
        m_locations.erase(pos);
        return true;
    }
    return false;
}


class BreakpointLocationIDMatches
{
public:
    BreakpointLocationIDMatches (lldb::break_id_t break_id) :
        m_break_id(break_id)
    {
    }

    bool operator() (const BreakpointLocationSP &bp_loc_sp) const
    {
        return m_break_id == bp_loc_sp->GetID();
    }

private:
   const lldb::break_id_t m_break_id;
};

class BreakpointLocationAddressMatches
{
public:
    BreakpointLocationAddressMatches (Address& addr) :
        m_addr(addr)
    {
    }

    bool operator() (const BreakpointLocationSP& bp_loc_sp) const
    {
        return Address::CompareFileAddress(m_addr, bp_loc_sp->GetAddress()) == 0;
    }

private:
   const Address &m_addr;
};

BreakpointLocationList::collection::iterator
BreakpointLocationList::GetIDIterator (lldb::break_id_t break_id)
{
    Mutex::Locker locker (m_mutex);
    return std::find_if (m_locations.begin(),
                         m_locations.end(),
                         BreakpointLocationIDMatches(break_id));
}

BreakpointLocationList::collection::const_iterator
BreakpointLocationList::GetIDConstIterator (lldb::break_id_t break_id) const
{
    Mutex::Locker locker (m_mutex);
    return std::find_if (m_locations.begin(),
                         m_locations.end(),
                         BreakpointLocationIDMatches(break_id));
}

BreakpointLocationSP
BreakpointLocationList::FindByID (lldb::break_id_t break_id)
{
    Mutex::Locker locker (m_mutex);
    BreakpointLocationSP stop_sp;
    collection::iterator pos = GetIDIterator(break_id);
    if (pos != m_locations.end())
        stop_sp = *pos;

    return stop_sp;
}

const BreakpointLocationSP
BreakpointLocationList::FindByID (lldb::break_id_t break_id) const
{
    Mutex::Locker locker (m_mutex);
    BreakpointLocationSP stop_sp;
    collection::const_iterator pos = GetIDConstIterator(break_id);
    if (pos != m_locations.end())
        stop_sp = *pos;

    return stop_sp;
}

size_t
BreakpointLocationList::FindInModule (Module *module,
                                      BreakpointLocationCollection& bp_loc_list)
{
    Mutex::Locker locker (m_mutex);
    const size_t orig_size = bp_loc_list.GetSize();
    collection::iterator pos, end = m_locations.end();

    for (pos = m_locations.begin(); pos != end; ++pos)
    {
        bool seen = false;
        BreakpointLocationSP break_loc = (*pos);
        const Section *section = break_loc->GetAddress().GetSection();
        if (section)
        {
            if (section->GetModule() == module)
            {
                if (!seen)
                {
                    seen = true;
                    bp_loc_list.Add (break_loc);
                }

            }
        }
    }
    return bp_loc_list.GetSize() - orig_size;
}

const BreakpointLocationSP
BreakpointLocationList::FindByAddress (Address &addr) const
{
    Mutex::Locker locker (m_mutex);
    BreakpointLocationSP stop_sp;
    if (!m_locations.empty())
    {
        addr_map::const_iterator pos = m_address_to_location.find (addr);
        if (pos != m_address_to_location.end())
            stop_sp = pos->second;
    }

    return stop_sp;
}

void
BreakpointLocationList::Dump (Stream *s) const
{
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    Mutex::Locker locker (m_mutex);
    s->Printf("BreakpointLocationList with %zu BreakpointLocations:\n", m_locations.size());
    s->IndentMore();
    collection::const_iterator pos, end = m_locations.end();
    for (pos = m_locations.begin(); pos != end; ++pos)
        (*pos).get()->Dump(s);
    s->IndentLess();
}


BreakpointLocationSP
BreakpointLocationList::GetByIndex (uint32_t i)
{
    Mutex::Locker locker (m_mutex);
    BreakpointLocationSP stop_sp;
    if (i < m_locations.size())
        stop_sp = m_locations[i];

    return stop_sp;
}

const BreakpointLocationSP
BreakpointLocationList::GetByIndex (uint32_t i) const
{
    Mutex::Locker locker (m_mutex);
    BreakpointLocationSP stop_sp;
    if (i < m_locations.size())
        stop_sp = m_locations[i];

    return stop_sp;
}

void
BreakpointLocationList::ClearAllBreakpointSites ()
{
    Mutex::Locker locker (m_mutex);
    collection::iterator pos, end = m_locations.end();
    for (pos = m_locations.begin(); pos != end; ++pos)
        (*pos)->ClearBreakpointSite();
}

void
BreakpointLocationList::ResolveAllBreakpointSites ()
{
    Mutex::Locker locker (m_mutex);
    collection::iterator pos, end = m_locations.end();

    for (pos = m_locations.begin(); pos != end; ++pos)
    {
        if ((*pos)->IsEnabled())
            (*pos)->ResolveBreakpointSite();
    }
}

uint32_t
BreakpointLocationList::GetHitCount () const
{
    uint32_t hit_count = 0;
    Mutex::Locker locker (m_mutex);
    collection::const_iterator pos, end = m_locations.end();
    for (pos = m_locations.begin(); pos != end; ++pos)
        hit_count += (*pos)->GetHitCount();
    return hit_count;
}

size_t
BreakpointLocationList::GetNumResolvedLocations() const
{
    Mutex::Locker locker (m_mutex);
    size_t resolve_count = 0;
    collection::const_iterator pos, end = m_locations.end();
    for (pos = m_locations.begin(); pos != end; ++pos)
    {
        if ((*pos)->IsResolved())
            ++resolve_count;
    }
    return resolve_count;
}

break_id_t
BreakpointLocationList::GetNextID()
{
    return ++m_next_id;
}

void
BreakpointLocationList::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    Mutex::Locker locker (m_mutex);
    collection::iterator pos, end = m_locations.end();

    for (pos = m_locations.begin(); pos != end; ++pos)
    {
        s->Printf(" ");
        (*pos)->GetDescription(s, level);
    }
}

