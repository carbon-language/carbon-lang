//===-- PathMappingList.cpp -------------------------------------*- C++ -*-===//
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
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
// Project includes
#include "PathMappingList.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// PathMappingList constructor
//----------------------------------------------------------------------
PathMappingList::PathMappingList 
(
    ChangedCallback callback,
    void *callback_baton
) :
    m_pairs (),
    m_callback (callback),
    m_callback_baton (callback_baton)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
PathMappingList::~PathMappingList ()
{
}

void
PathMappingList::Append (const ConstString &path,
                         const ConstString &replacement,
                         bool notify)
{
    m_pairs.push_back(pair(path, replacement));
    if (notify && m_callback)
        m_callback (*this, m_callback_baton);
}

void
PathMappingList::Insert (const ConstString &path,
                         const ConstString &replacement,
                         uint32_t index,
                         bool notify)
{
    iterator insert_iter;
    if (index >= m_pairs.size())
        insert_iter = m_pairs.end();
    else
        insert_iter = m_pairs.begin() + index;
    m_pairs.insert(insert_iter, pair(path, replacement));
    if (notify && m_callback)
        m_callback (*this, m_callback_baton);
}

bool
PathMappingList::Remove (off_t index, bool notify)
{
    if (index >= m_pairs.size())
        return false;

    iterator iter = m_pairs.begin() + index;
    m_pairs.erase(iter);
    if (notify && m_callback)
        m_callback (*this, m_callback_baton);
    return true;
}

void
PathMappingList::Dump (Stream *s)
{
    unsigned int numPairs = m_pairs.size();
    unsigned int index;

    for (index = 0; index < numPairs; ++index)
    {
        s->Printf("[%d] \"%s\" -> \"%s\"\n",
                  index, m_pairs[index].first.GetCString(), m_pairs[index].second.GetCString());
    }
}

void
PathMappingList::Clear (bool notify)
{
    m_pairs.clear();
    if (notify && m_callback)
        m_callback (*this, m_callback_baton);
}

size_t
PathMappingList::GetSize ()
{
    return m_pairs.size();
}

bool
PathMappingList::RemapPath (const ConstString &path, ConstString &new_path)
{
    const_iterator pos, end = m_pairs.end();
    for (pos = m_pairs.begin(); pos != end; ++pos)
    {
        const size_t prefixLen = pos->first.GetLength();

        if (::strncmp (pos->first.GetCString(), path.GetCString(), prefixLen) == 0)
        {
            std::string new_path_str (pos->second.GetCString());
            new_path_str.append(path.GetCString() + prefixLen);
            new_path.SetCString(new_path_str.c_str());
            return true;
        }
    }
    return false;
}
