//===-- UniqueCStringMap.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_UniqueCStringMap_h_
#define liblldb_UniqueCStringMap_h_
#if defined(__cplusplus)

#include <assert.h>
#include <algorithm>
#include <vector>

namespace lldb_private {



//----------------------------------------------------------------------
// Templatized uniqued string map.
//
// This map is useful for mapping unique C string names to values of
// type T. Each "const char *" name added must be unique for a given
// C string value. ConstString::GetCString() can provide such strings.
// Any other string table that has guaranteed unique values can also
// be used.
//----------------------------------------------------------------------
template <typename T>
class UniqueCStringMap
{
public:
    struct Entry
    {
        Entry () :
            cstring(NULL),
            value()
        {
        }

        Entry (const char *cstr) :
            cstring(cstr),
            value()
        {
        }

        Entry (const char *cstr, const T&v) :
            cstring(cstr),
            value(v)
        {
        }

        bool
        operator < (const Entry& rhs) const
        {
            return cstring < rhs.cstring;
        }

        const char* cstring;
        T value;
    };

    //------------------------------------------------------------------
    // Call this function multiple times to add a bunch of entries to
    // this map, then later call UniqueCStringMap<T>::Sort() before doing
    // any searches by name.
    //------------------------------------------------------------------
    void
    Append (const char *unique_cstr, const T& value)
    {
        m_map.push_back (typename UniqueCStringMap<T>::Entry(unique_cstr, value));
    }

    void
    Append (const Entry &e)
    {
        m_map.push_back (e);
    }

    void
    Clear ()
    {
        m_map.clear();
    }

    //------------------------------------------------------------------
    // Call this function to always keep the map sorted when putting
    // entries into the map.
    //------------------------------------------------------------------
    void
    Insert (const char *unique_cstr, const T& value)
    {
        typename UniqueCStringMap<T>::Entry e(unique_cstr, value);
        m_map.insert (std::upper_bound (m_map.begin(), m_map.end(), e), e);
    }

    void
    Insert (const Entry &e)
    {
        m_map.insert (std::upper_bound (m_map.begin(), m_map.end(), e), e);
    }

    //------------------------------------------------------------------
    // Get an entry by index.
    //
    // The caller is responsible for ensuring that the collection does
    // not change during while using the returned pointer.
    //------------------------------------------------------------------
    const T *
    GetValueAtIndex (uint32_t idx) const
    {
        if (idx < m_map.size())
            return &m_map[idx].value;
        return NULL;
    }

    const char *
    GetCStringAtIndex (uint32_t idx) const
    {
        if (idx < m_map.size())
            return m_map[idx].cstring;
        return NULL;
    }

    //------------------------------------------------------------------
    // Get a pointer to the first entry that matches "name". NULL will
    // be returned if there is no entry that matches "name".
    //
    // The caller is responsible for ensuring that the collection does
    // not change during while using the returned pointer.
    //------------------------------------------------------------------
    const Entry *
    FindFirstValueForName (const char *unique_cstr) const
    {
        Entry search_entry (unique_cstr);
        const_iterator end = m_map.end();
        const_iterator pos = std::lower_bound (m_map.begin(), end, search_entry);
        if (pos != end)
        {
            const char *pos_cstr = pos->cstring;
            if (pos_cstr == unique_cstr)
                return &(*pos);
        }
        return NULL;
    }

    //------------------------------------------------------------------
    // Get a pointer to the next entry that matches "name" from a
    // previously returned Entry pointer. NULL will be returned if there
    // is no subsequent entry that matches "name".
    //
    // The caller is responsible for ensuring that the collection does
    // not change during while using the returned pointer.
    //------------------------------------------------------------------
    const Entry *
    FindNextValueForName (const char *unique_cstr, const Entry *entry_ptr) const
    {
        if (!m_map.empty())
        {
            const Entry *first_entry = &m_map[0];
            const Entry *after_last_entry = first_entry + m_map.size();
            const Entry *next_entry = entry_ptr + 1;
            if (first_entry <= next_entry && next_entry < after_last_entry)
            {
                if (next_entry->cstring == unique_cstr)
                    return next_entry;
            }
        }
        return NULL;
    }

    //------------------------------------------------------------------
    // Get the total number of entries in this map.
    //------------------------------------------------------------------
    size_t
    GetSize () const
    {
        return m_map.size();
    }


    //------------------------------------------------------------------
    // Returns true if this map is empty.
    //------------------------------------------------------------------
    bool
    IsEmpty() const
    {
        return m_map.empty();
    }

    //------------------------------------------------------------------
    // Reserve memory for at least "n" entries in the map. This is
    // useful to call when you know you will be adding a lot of entries
    // using UniqueCStringMap::Append() (which should be followed by a
    // call to UniqueCStringMap::Sort()) or to UniqueCStringMap::Insert().
    //------------------------------------------------------------------
    void
    Reserve (size_t n)
    {
        m_map.reserve (n);
    }

    //------------------------------------------------------------------
    // Sort the unsorted contents in this map. A typical code flow would
    // be:
    // size_t approximate_num_entries = ....
    // UniqueCStringMap<uint32_t> my_map;
    // my_map.Reserve (approximate_num_entries);
    // for (...)
    // {
    //      my_map.Append (UniqueCStringMap::Entry(GetName(...), GetValue(...)));
    // }
    // my_map.Sort();
    //------------------------------------------------------------------
    void
    Sort ()
    {
        std::sort (m_map.begin(), m_map.end());
    }

protected:
    typedef std::vector<Entry> collection;
    typedef typename collection::iterator iterator;
    typedef typename collection::const_iterator const_iterator;
    collection m_map;
};



} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_UniqueCStringMap_h_
