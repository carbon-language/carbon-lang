//===-- WatchpointList.cpp --------------------------------------*- C++ -*-===//
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
#include "lldb/Breakpoint/WatchpointList.h"
#include "lldb/Breakpoint/Watchpoint.h"

using namespace lldb;
using namespace lldb_private;

WatchpointList::WatchpointList() :
    m_address_to_watchpoint (),
    m_mutex (Mutex::eMutexTypeRecursive)
{
}

WatchpointList::~WatchpointList()
{
}

// Add watchpoint loc to the list.  However, if the element already exists in the
// list, then replace it with the input one.

lldb::watch_id_t
WatchpointList::Add (const WatchpointSP &wp_sp)
{
    Mutex::Locker locker (m_mutex);
    lldb::addr_t wp_addr = wp_sp->GetLoadAddress();
    addr_map::iterator iter = m_address_to_watchpoint.find(wp_addr);

    if (iter == m_address_to_watchpoint.end())
        m_address_to_watchpoint.insert(iter, addr_map::value_type(wp_addr, wp_sp));
    else
        m_address_to_watchpoint[wp_addr] = wp_sp;

    return wp_sp->GetID();
}

void
WatchpointList::Dump (Stream *s) const
{
    DumpWithLevel(s, lldb::eDescriptionLevelBrief);
}

void
WatchpointList::DumpWithLevel (Stream *s, lldb::DescriptionLevel description_level) const
{
    Mutex::Locker locker (m_mutex);
    s->Printf("%p: ", this);
    //s->Indent();
    s->Printf("WatchpointList with %zu Watchpoints:\n",
              m_address_to_watchpoint.size());
    s->IndentMore();
    addr_map::const_iterator pos, end = m_address_to_watchpoint.end();
    for (pos = m_address_to_watchpoint.begin(); pos != end; ++pos)
        pos->second->DumpWithLevel(s, description_level);
    s->IndentLess();
}

const WatchpointSP
WatchpointList::FindByAddress (lldb::addr_t addr) const
{
    WatchpointSP wp_sp;
    Mutex::Locker locker (m_mutex);
    if (!m_address_to_watchpoint.empty())
    {
        addr_map::const_iterator pos = m_address_to_watchpoint.find (addr);
        if (pos != m_address_to_watchpoint.end())
            wp_sp = pos->second;
    }

    return wp_sp;
}

class WatchpointIDMatches
{
public:
    WatchpointIDMatches (lldb::watch_id_t watch_id) :
        m_watch_id(watch_id)
    {
    }

    bool operator() (std::pair <lldb::addr_t, WatchpointSP> val_pair) const
    {
        return m_watch_id == val_pair.second.get()->GetID();
    }

private:
   const lldb::watch_id_t m_watch_id;
};

WatchpointList::addr_map::iterator
WatchpointList::GetIDIterator (lldb::watch_id_t watch_id)
{
    return std::find_if(m_address_to_watchpoint.begin(), m_address_to_watchpoint.end(), // Search full range
                        WatchpointIDMatches(watch_id));                                 // Predicate
}

WatchpointList::addr_map::const_iterator
WatchpointList::GetIDConstIterator (lldb::watch_id_t watch_id) const
{
    return std::find_if(m_address_to_watchpoint.begin(), m_address_to_watchpoint.end(), // Search full range
                        WatchpointIDMatches(watch_id));                                 // Predicate
}

WatchpointSP
WatchpointList::FindByID (lldb::watch_id_t watch_id) const
{
    WatchpointSP wp_sp;
    Mutex::Locker locker (m_mutex);
    addr_map::const_iterator pos = GetIDConstIterator(watch_id);
    if (pos != m_address_to_watchpoint.end())
        wp_sp = pos->second;

    return wp_sp;
}

lldb::watch_id_t
WatchpointList::FindIDByAddress (lldb::addr_t addr)
{
    WatchpointSP wp_sp = FindByAddress (addr);
    if (wp_sp)
    {
        return wp_sp->GetID();
    }
    return LLDB_INVALID_WATCH_ID;
}

WatchpointSP
WatchpointList::GetByIndex (uint32_t i)
{
    Mutex::Locker locker (m_mutex);
    WatchpointSP wp_sp;
    if (i < m_address_to_watchpoint.size())
    {
        addr_map::const_iterator pos = m_address_to_watchpoint.begin();
        std::advance(pos, i);
        wp_sp = pos->second;
    }
    return wp_sp;
}

const WatchpointSP
WatchpointList::GetByIndex (uint32_t i) const
{
    Mutex::Locker locker (m_mutex);
    WatchpointSP wp_sp;
    if (i < m_address_to_watchpoint.size())
    {
        addr_map::const_iterator pos = m_address_to_watchpoint.begin();
        std::advance(pos, i);
        wp_sp = pos->second;
    }
    return wp_sp;
}

bool
WatchpointList::Remove (lldb::watch_id_t watch_id)
{
    Mutex::Locker locker (m_mutex);
    addr_map::iterator pos = GetIDIterator(watch_id);
    if (pos != m_address_to_watchpoint.end())
    {
        m_address_to_watchpoint.erase(pos);
        return true;
    }
    return false;
}

uint32_t
WatchpointList::GetHitCount () const
{
    uint32_t hit_count = 0;
    Mutex::Locker locker (m_mutex);
    addr_map::const_iterator pos, end = m_address_to_watchpoint.end();
    for (pos = m_address_to_watchpoint.begin(); pos != end; ++pos)
        hit_count += pos->second->GetHitCount();
    return hit_count;
}

bool
WatchpointList::ShouldStop (StoppointCallbackContext *context, lldb::watch_id_t watch_id)
{

    WatchpointSP wp_sp = FindByID (watch_id);
    if (wp_sp)
    {
        // Let the Watchpoint decide if it should stop here (could not have
        // reached it's target hit count yet, or it could have a callback
        // that decided it shouldn't stop.
        return wp_sp->ShouldStop (context);
    }
    // We should stop here since this Watchpoint isn't valid anymore or it
    // doesn't exist.
    return true;
}

void
WatchpointList::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    Mutex::Locker locker (m_mutex);
    addr_map::iterator pos, end = m_address_to_watchpoint.end();

    for (pos = m_address_to_watchpoint.begin(); pos != end; ++pos)
    {
        s->Printf(" ");
        pos->second->Dump(s);
    }
}

void
WatchpointList::SetEnabledAll (bool enabled)
{
    Mutex::Locker locker(m_mutex);

    addr_map::iterator pos, end = m_address_to_watchpoint.end();
    for (pos = m_address_to_watchpoint.begin(); pos != end; ++pos)
        pos->second->SetEnabled (enabled);
}

void
WatchpointList::RemoveAll ()
{
    Mutex::Locker locker(m_mutex);
    m_address_to_watchpoint.clear();
}

void
WatchpointList::GetListMutex (Mutex::Locker &locker)
{
    return locker.Reset (m_mutex.GetMutex());
}
