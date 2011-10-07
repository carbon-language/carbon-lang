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
    // Templatized classes for dealing with generic ranges and also 
    // collections of ranges, or collections of ranges that have associated
    // data.
    //----------------------------------------------------------------------
    
    //----------------------------------------------------------------------
    // A simple range class where you get to define the type of the range
    // base "B", and the type used for the range byte size "S".
    //----------------------------------------------------------------------
    template <typename B, typename S>
    struct Range
    {
        typedef B BaseType;
        typedef S SizeType;

        BaseType base;
        SizeType size;
        
        Range () :
            base (0),
            size (0)
        {
        }
        
        Range (BaseType b, SizeType s) :
            base (b),
            size (s)
        {
        }
        
        // Set the start value for the range, and keep the same size
        BaseType
        GetRangeBase () const
        {
            return base;
        }
        
        void
        SetRangeBase (BaseType b)
        {
            base = b;
        }
        
        BaseType
        GetRangeEnd () const
        {
            return base + size;
        }
        
        void
        SetRangeEnd (BaseType end)
        {
            if (end > base)
                size = end - base;
            else
                size = 0;
        }
        
        SizeType
        GetByteSize () const
        {
            return size;
        }
        
        void
        SetByteSize (SizeType s)
        {
            size = s;
        }
        
        bool
        IsValid() const
        {
            return size > 0;
        }
        
        bool
        Contains (BaseType r) const
        {
            return (GetRangeBase() <= r) && (r < GetRangeEnd());
        }
        
        bool
        ContainsEndInclusive (BaseType r) const
        {
            return (GetRangeBase() <= r) && (r <= GetRangeEnd());
        }
        
        bool 
        Contains (const Range& range) const
        {
            return Contains(range.GetRangeBase()) && ContainsEndInclusive(range.GetRangeEnd());
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
    
    //----------------------------------------------------------------------
    // A range array class where you get to define the type of the ranges
    // that the collection contains.
    //----------------------------------------------------------------------

    template <typename B, typename S>
    class RangeArray
    {
        typedef Range<B,S> Entry;
        
        RangeArray ()
        {
        }
        
        ~RangeArray()
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
                        minimal_ranges.back().range.SetEnd (pos->range.GetRangeEnd());
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
            return lhs.GetRangeBase() < rhs.GetRangeBase();
        }
        
        const Entry *
        FindEntryThatContains (B addr) const
        {
            if ( !m_entries.empty() )
            {
                Entry entry (addr, 1);
                typename std::vector<Entry>::const_iterator begin = m_entries.begin();
                typename std::vector<Entry>::const_iterator end = m_entries.end();
                typename std::vector<Entry>::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                if (pos != end && pos->Contains(addr))
                {
                    return &(*pos); 
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(addr))
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

    //----------------------------------------------------------------------
    // A simple range  with data class where you get to define the type of
    // the range base "B", the type used for the range byte size "S", and
    // the type for the associated data "T".
    //----------------------------------------------------------------------
    template <typename B, typename S, typename T>
    struct RangeData : public Range<B,S>
    {
        typedef T DataType;
        
        DataType data;
        
        RangeData () :
            Range<B,S> (),
            data ()
        {
        }
        
        RangeData (B base, S size, DataType d) :
            Range<B,S> (base, size),
            data (d)
        {
        }
        
        bool
        operator < (const RangeData &rhs) const
        {
            if (this->base == rhs.base)
            {
                if (this->size == rhs.size)
                    return this->data < rhs.data;
                else
                    return this->size < rhs.size;
            }
            return this->base < rhs.base;
        }
        
        bool
        operator == (const RangeData &rhs) const
        {
            return this->GetRangeBase() == rhs.GetRangeBase() &&
                   this->GetByteSize() == rhs.GetByteSize() &&
                   this->data      == rhs.data;
        }
        
        bool
        operator != (const RangeData &rhs) const
        {
            return this->GetRangeBase() != rhs.GetRangeBase() ||
                   this->GetByteSize() != rhs.GetByteSize() ||
                   this->data      != rhs.data;
        }
    };
    
    template <typename B, typename S, typename T>
    class RangeDataArray
    {
    public:
        typedef RangeData<B,S,T> Entry;
        
        RangeDataArray ()
        {
        }
        
        ~RangeDataArray()
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
                        minimal_ranges.back().SetRangeEnd (pos->GetRangeEnd());
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
            return lhs.GetRangeBase() < rhs.GetRangeBase();
        }
        
        const Entry *
        FindEntryThatContains (B addr) const
        {
            if ( !m_entries.empty() )
            {
                Entry entry;
                entry.SetRangeBase(addr);
                entry.SetByteSize(1);
                typename std::vector<Entry>::const_iterator begin = m_entries.begin();
                typename std::vector<Entry>::const_iterator end = m_entries.end();
                typename std::vector<Entry>::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                if (pos != end && pos->Contains(addr))
                {
                    return &(*pos); 
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(addr))
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
