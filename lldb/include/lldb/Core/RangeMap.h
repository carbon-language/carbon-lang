//===-- RangeMap.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RangeMap_h_
#define liblldb_RangeMap_h_

#include "lldb/lldb-private.h"
#include <vector>

namespace lldb_private {

//----------------------------------------------------------------------
// A vm address range. These can represent offsets ranges or actual
// addresses.
//----------------------------------------------------------------------
template <typename B, typename S, class T>
class RangeMap 
{
public:
    typedef B RangeBaseType;
    typedef S RangeSizeType;
    typedef T EntryDataType;

    struct Range
    {
        RangeBaseType base;
        RangeSizeType size;
        
        Range () :
            base (0),
            size (0)
        {
        }
        
        Range (RangeBaseType b, RangeSizeType s) :
            base (b),
            size (s)
        {
        }
        
        // Set the start value for the range, and keep the same size
        RangeBaseType
        GetBase () const
        {
            return base;
        }
        
        void
        SetBase (RangeBaseType b)
        {
            base = b;
        }
        
        RangeBaseType
        GetEnd () const
        {
            return base + size;
        }
        
        void
        SetEnd (RangeBaseType end)
        {
            if (end > base)
                size = end - base;
            else
                size = 0;
        }
        
        RangeSizeType
        GetSize () const
        {
            return size;
        }
        
        void
        SetSize (RangeSizeType s)
        {
            size = s;
        }
        
        bool
        IsValid() const
        {
            return size > 0;
        }
        
        bool
        Contains (RangeBaseType r) const
        {
            return (GetBase() <= r) && (r < GetEnd());
        }
        
        bool
        ContainsEndInclusive (RangeBaseType r) const
        {
            return (GetBase() <= r) && (r <= GetEnd());
        }
        
        bool 
        Contains (const Range& range) const
        {
            return Contains(range.GetBase()) && ContainsEndInclusive(range.GetEnd());
        }
        
        bool
        operator < (const Range &rhs)
        {
            if (base == rhs.base)
                return size < rhs.size;
            return base < rhs.base;
        }
        
        bool
        operator == (const Range &rhs)
        {
            return base == rhs.base && size == rhs.size;
        }
        
        bool
        operator != (const Range &rhs)
        {
            return  base != rhs.base || size != rhs.size;
        }
    };

    struct Entry
    {
        Range range;
        EntryDataType data;

        Entry () :
            range (),
            data ()
        {
        }

        Entry (RangeBaseType base, RangeSizeType size, EntryDataType d) :
            range (base, size),
            data (d)
        {
        }

        bool
        operator < (const Entry &rhs) const
        {
            const RangeBaseType lhs_base = range.GetBase();
            const RangeBaseType rhs_base = rhs.range.GetBase();
            if (lhs_base == rhs_base)
            {
                const RangeBaseType lhs_size = range.GetSize();
                const RangeBaseType rhs_size = rhs.range.GetSize();
                if (lhs_size == rhs_size)
                    return data < rhs.data;
                else
                    return lhs_size < rhs_size;
            }
            return lhs_base < rhs_base;
        }

        bool
        operator == (const Entry &rhs) const
        {
            return range.GetBase() == rhs.range.GetBase() &&
                   range.GetSize() == rhs.range.GetSize() &&
                   data == rhs.data;
        }
    
        bool
        operator != (const Entry &rhs) const
        {
            return  range.GetBase() != rhs.range.GetBase() ||
                    range.GetSize() != rhs.range.GetSize() ||
                    data != rhs.data;
        }
    };

    RangeMap ()
    {
    }

    ~RangeMap()
    {
    }

    void
    Append (const Entry &entry)
    {
        m_entries.push_back (entry);
    }

    void
    Sort ()
    {
        if (m_entries.size() > 1)
            std::stable_sort (m_entries.begin(), m_entries.end());
    }
    
    void
    CombineConsecutiveEntriesWithEqualData ()
    {
        typename std::vector<Entry>::iterator pos;
        typename std::vector<Entry>::iterator end;
        typename std::vector<Entry>::iterator prev;
        bool can_combine = false;
        // First we determine if we can combine any of the Entry objects so we
        // don't end up allocating and making a new collection for no reason
        for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
        {
            if (prev != end && prev->data == pos->data)
            {
                can_combine = true;
                break;
            }
        }
        
        // We we can combine at least one entry, then we make a new collection
        // and populate it accordingly, and then swap it into place. 
        if (can_combine)
        {
            std::vector<Entry> minimal_ranges;
            for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            {
                if (prev != end && prev->data == pos->data)
                    minimal_ranges.back().range.SetEnd (pos->range.GetEnd());
                else
                    minimal_ranges.push_back (*pos);
            }
            // Use the swap technique in case our new vector is much smaller.
            // We must swap when using the STL because std::vector objects never
            // release or reduce the memory once it has been allocated/reserved.
            m_entries.swap (minimal_ranges);
        }
    }

    void
    Clear ()
    {
        m_entries.clear();
    }

    bool
    IsEmpty () const
    {
        return m_entries.empty();
    }

    size_t
    GetNumEntries () const
    {
        return m_entries.size();
    }

    const Entry *
    GetEntryAtIndex (uint32_t i) const
    {
        if (i<m_entries.size())
            return &m_entries[i];
        return NULL;
    }

    static bool 
    BaseLessThan (const Entry& lhs, const Entry& rhs)
    {
        return lhs.range.GetBase() < rhs.range.GetBase();
    }

    const Entry *
    FindEntryThatContains (RangeBaseType addr) const
    {
        if ( !m_entries.empty() )
        {
            Entry entry;
            entry.range.SetBase(addr);
            typename std::vector<Entry>::const_iterator begin = m_entries.begin();
            typename std::vector<Entry>::const_iterator end = m_entries.end();
            typename std::vector<Entry>::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
            
            if ((pos != end) && (pos->range.GetBase() <= addr && addr < pos->range.GetEnd()))
            {
                return &(*pos); 
            }
            else if (pos != begin)
            {
                --pos;
                if ((pos->range.GetBase() <= addr) && (addr < pos->range.GetEnd()))
                {
                    return &(*pos); 
                }
            }
        }
        return NULL;
    }

protected:
    std::vector<Entry> m_entries;
};


} // namespace lldb_private

#endif  // liblldb_RangeMap_h_
