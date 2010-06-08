//===-- BreakpointList.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointList.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

BreakpointList::BreakpointList (bool is_internal) :
    m_mutex (Mutex::eMutexTypeRecursive),
    m_breakpoints(),
    m_next_break_id (0),
    m_is_internal (is_internal)
{
}

BreakpointList::~BreakpointList()
{
}


break_id_t
BreakpointList::Add (BreakpointSP &bp)
{
    Mutex::Locker locker(m_mutex);
    // Internal breakpoint IDs are negative, normal ones are positive
    bp->SetID (m_is_internal ? --m_next_break_id : ++m_next_break_id);
    m_breakpoints.push_back(bp);
    return bp->GetID();
}

bool
BreakpointList::Remove (break_id_t break_id)
{
    Mutex::Locker locker(m_mutex);
    bp_collection::iterator pos = GetBreakpointIDIterator(break_id);    // Predicate
    if (pos != m_breakpoints.end())
    {
        m_breakpoints.erase(pos);
        return true;
    }
    return false;
}

void
BreakpointList::SetEnabledAll (bool enabled)
{
    Mutex::Locker locker(m_mutex);
    bp_collection::iterator pos, end = m_breakpoints.end();
    for (pos = m_breakpoints.begin(); pos != end; ++pos)
        (*pos)->SetEnabled (enabled);
}


void
BreakpointList::RemoveAll ()
{
    Mutex::Locker locker(m_mutex);
    ClearAllBreakpointSites ();

    m_breakpoints.erase (m_breakpoints.begin(), m_breakpoints.end());
}

class BreakpointIDMatches
{
public:
    BreakpointIDMatches (break_id_t break_id) :
        m_break_id(break_id)
    {
    }

    bool operator() (const BreakpointSP &bp) const
    {
        return m_break_id == bp->GetID();
    }

private:
   const break_id_t m_break_id;
};

BreakpointList::bp_collection::iterator
BreakpointList::GetBreakpointIDIterator (break_id_t break_id)
{
    return std::find_if(m_breakpoints.begin(), m_breakpoints.end(), // Search full range
                        BreakpointIDMatches(break_id));             // Predicate
}

BreakpointList::bp_collection::const_iterator
BreakpointList::GetBreakpointIDConstIterator (break_id_t break_id) const
{
    return std::find_if(m_breakpoints.begin(), m_breakpoints.end(), // Search full range
                        BreakpointIDMatches(break_id));             // Predicate
}

BreakpointSP
BreakpointList::FindBreakpointByID (break_id_t break_id)
{
    Mutex::Locker locker(m_mutex);
    BreakpointSP stop_sp;
    bp_collection::iterator pos = GetBreakpointIDIterator(break_id);
    if (pos != m_breakpoints.end())
        stop_sp = *pos;

    return stop_sp;
}

const BreakpointSP
BreakpointList::FindBreakpointByID (break_id_t break_id) const
{
    Mutex::Locker locker(m_mutex);
    BreakpointSP stop_sp;
    bp_collection::const_iterator pos = GetBreakpointIDConstIterator(break_id);
    if (pos != m_breakpoints.end())
        stop_sp = *pos;

    return stop_sp;
}

void
BreakpointList::Dump (Stream *s) const
{
    Mutex::Locker locker(m_mutex);
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    s->Printf("BreakpointList with %u Breakpoints:\n", (uint32_t)m_breakpoints.size());
    s->IndentMore();
    bp_collection::const_iterator pos;
    bp_collection::const_iterator end = m_breakpoints.end();
    for (pos = m_breakpoints.begin(); pos != end; ++pos)
        (*pos)->Dump(s);
    s->IndentLess();
}


BreakpointSP
BreakpointList::GetBreakpointByIndex (uint32_t i)
{
    Mutex::Locker locker(m_mutex);
    BreakpointSP stop_sp;
    bp_collection::iterator end = m_breakpoints.end();
    bp_collection::iterator pos;
    uint32_t curr_i = 0;
    for (pos = m_breakpoints.begin(), curr_i = 0; pos != end; ++pos, ++curr_i)
    {
        if (curr_i == i)
            stop_sp = *pos;
    }
    return stop_sp;
}

const BreakpointSP
BreakpointList::GetBreakpointByIndex (uint32_t i) const
{
    Mutex::Locker locker(m_mutex);
    BreakpointSP stop_sp;
    bp_collection::const_iterator end = m_breakpoints.end();
    bp_collection::const_iterator pos;
    uint32_t curr_i = 0;
    for (pos = m_breakpoints.begin(), curr_i = 0; pos != end; ++pos, ++curr_i)
    {
        if (curr_i == i)
            stop_sp = *pos;
    }
    return stop_sp;
}

void
BreakpointList::UpdateBreakpoints (ModuleList& module_list, bool added)
{
    Mutex::Locker locker(m_mutex);
    bp_collection::iterator end = m_breakpoints.end();
    bp_collection::iterator pos;
    for (pos = m_breakpoints.begin(); pos != end; ++pos)
        (*pos)->ModulesChanged (module_list, added);

}

void
BreakpointList::ClearAllBreakpointSites ()
{
    Mutex::Locker locker(m_mutex);
    bp_collection::iterator end = m_breakpoints.end();
    bp_collection::iterator pos;
    for (pos = m_breakpoints.begin(); pos != end; ++pos)
        (*pos)->ClearAllBreakpointSites ();

}
