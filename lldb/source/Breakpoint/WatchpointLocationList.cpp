//===-- WatchpointLocationList.cpp ------------------------------*- C++ -*-===//
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
#include "lldb/Breakpoint/WatchpointLocationList.h"
#include "lldb/Breakpoint/WatchpointLocation.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

WatchpointLocationList::WatchpointLocationList() :
    m_address_to_location (),
    m_mutex (Mutex::eMutexTypeRecursive)
{
}

WatchpointLocationList::~WatchpointLocationList()
{
}

// Add watchpoint loc to the list.  However, if the element already exists in the
// list, then replace it with the input one.

lldb::watch_id_t
WatchpointLocationList::Add (const WatchpointLocationSP &wp_loc_sp)
{
    Mutex::Locker locker (m_mutex);
    lldb::addr_t wp_addr = wp_loc_sp->GetLoadAddress();
    addr_map::iterator iter = m_address_to_location.find(wp_addr);

    if (iter == m_address_to_location.end())
    {
        m_address_to_location.insert(iter, addr_map::value_type(wp_addr, wp_loc_sp));
    }
    else
    {
        m_address_to_location[wp_addr] = wp_loc_sp;
    }
    return wp_loc_sp->GetID();
}

void
WatchpointLocationList::Dump (Stream *s) const
{
    Mutex::Locker locker (m_mutex);
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    //s->Indent();
    s->Printf("WatchpointLocationList with %zu WatchpointLocations:\n",
              m_address_to_location.size());
    s->IndentMore();
    addr_map::const_iterator pos, end = m_address_to_location.end();
    for (pos = m_address_to_location.begin(); pos != end; ++pos)
        pos->second->Dump(s);
    s->IndentLess();
}

const WatchpointLocationSP
WatchpointLocationList::FindByAddress (lldb::addr_t addr) const
{
    WatchpointLocationSP wp_loc_sp;
    Mutex::Locker locker (m_mutex);
    if (!m_address_to_location.empty())
    {
        addr_map::const_iterator pos = m_address_to_location.find (addr);
        if (pos != m_address_to_location.end())
            wp_loc_sp = pos->second;
    }

    return wp_loc_sp;
}

class WatchpointLocationIDMatches
{
public:
    WatchpointLocationIDMatches (lldb::watch_id_t watch_id) :
        m_watch_id(watch_id)
    {
    }

    bool operator() (std::pair <lldb::addr_t, WatchpointLocationSP> val_pair) const
    {
        return m_watch_id == val_pair.second.get()->GetID();
    }

private:
   const lldb::watch_id_t m_watch_id;
};

WatchpointLocationList::addr_map::iterator
WatchpointLocationList::GetIDIterator (lldb::watch_id_t watch_id)
{
    return std::find_if(m_address_to_location.begin(), m_address_to_location.end(), // Search full range
                        WatchpointLocationIDMatches(watch_id));                     // Predicate
}

WatchpointLocationList::addr_map::const_iterator
WatchpointLocationList::GetIDConstIterator (lldb::watch_id_t watch_id) const
{
    return std::find_if(m_address_to_location.begin(), m_address_to_location.end(), // Search full range
                        WatchpointLocationIDMatches(watch_id));                     // Predicate
}

WatchpointLocationSP
WatchpointLocationList::FindByID (lldb::watch_id_t watch_id) const
{
    WatchpointLocationSP wp_loc_sp;
    Mutex::Locker locker (m_mutex);
    addr_map::const_iterator pos = GetIDConstIterator(watch_id);
    if (pos != m_address_to_location.end())
        wp_loc_sp = pos->second;

    return wp_loc_sp;
}

lldb::watch_id_t
WatchpointLocationList::FindIDByAddress (lldb::addr_t addr)
{
    WatchpointLocationSP wp_loc_sp = FindByAddress (addr);
    if (wp_loc_sp)
    {
        return wp_loc_sp->GetID();
    }
    return LLDB_INVALID_WATCH_ID;
}

bool
WatchpointLocationList::Remove (lldb::watch_id_t watch_id)
{
    Mutex::Locker locker (m_mutex);
    addr_map::iterator pos = GetIDIterator(watch_id);    // Predicate
    if (pos != m_address_to_location.end())
    {
        m_address_to_location.erase(pos);
        return true;
    }
    return false;
}

uint32_t
WatchpointLocationList::GetHitCount () const
{
    uint32_t hit_count = 0;
    Mutex::Locker locker (m_mutex);
    addr_map::const_iterator pos, end = m_address_to_location.end();
    for (pos = m_address_to_location.begin(); pos != end; ++pos)
        hit_count += pos->second->GetHitCount();
    return hit_count;
}

bool
WatchpointLocationList::ShouldStop (StoppointCallbackContext *context, lldb::watch_id_t watch_id)
{
    WatchpointLocationSP wp_loc_sp = FindByID (watch_id);
    if (wp_loc_sp)
    {
        // Let the WatchpointLocation decide if it should stop here (could not have
        // reached it's target hit count yet, or it could have a callback
        // that decided it shouldn't stop.
        return wp_loc_sp->ShouldStop (context);
    }
    // We should stop here since this WatchpointLocation isn't valid anymore or it
    // doesn't exist.
    return true;
}

void
WatchpointLocationList::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    Mutex::Locker locker (m_mutex);
    addr_map::iterator pos, end = m_address_to_location.end();

    for (pos = m_address_to_location.begin(); pos != end; ++pos)
    {
        s->Printf(" ");
        pos->second->Dump(s);
    }
}

void
WatchpointLocationList::GetListMutex (Mutex::Locker &locker)
{
    return locker.Reset (m_mutex.GetMutex());
}
