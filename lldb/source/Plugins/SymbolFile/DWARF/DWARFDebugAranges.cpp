//===-- DWARFDebugAranges.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugAranges.h"

#include <assert.h>
#include <stdio.h>

#include <algorithm>

#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"

#include "SymbolFileDWARF.h"
#include "DWARFDebugInfo.h"
#include "DWARFCompileUnit.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
DWARFDebugAranges::DWARFDebugAranges() :
    m_aranges()
{
}


//----------------------------------------------------------------------
// Compare function DWARFDebugAranges::Range structures
//----------------------------------------------------------------------
static bool RangeLessThan (const DWARFDebugAranges::Range& range1, const DWARFDebugAranges::Range& range2)
{
    return range1.lo_pc < range2.lo_pc;
}

//----------------------------------------------------------------------
// CountArangeDescriptors
//----------------------------------------------------------------------
class CountArangeDescriptors
{
public:
    CountArangeDescriptors (uint32_t& count_ref) : count(count_ref)
    {
//      printf("constructor CountArangeDescriptors()\n");
    }
    void operator() (const DWARFDebugArangeSet& set)
    {
        count += set.NumDescriptors();
    }
    uint32_t& count;
};

//----------------------------------------------------------------------
// AddArangeDescriptors
//----------------------------------------------------------------------
class AddArangeDescriptors
{
public:
    AddArangeDescriptors (DWARFDebugAranges::RangeColl& ranges) : range_collection(ranges) {}
    void operator() (const DWARFDebugArangeSet& set)
    {
        const DWARFDebugArangeSet::Descriptor* arange_desc_ptr;
        DWARFDebugAranges::Range range;
        range.offset = set.GetCompileUnitDIEOffset();

        for (uint32_t i=0; (arange_desc_ptr = set.GetDescriptor(i)) != NULL; ++i)
        {
            range.lo_pc = arange_desc_ptr->address;
            range.hi_pc = arange_desc_ptr->address + arange_desc_ptr->length;

            // Insert each item in increasing address order so binary searching
            // can later be done!
            DWARFDebugAranges::RangeColl::iterator insert_pos = lower_bound(range_collection.begin(), range_collection.end(), range, RangeLessThan);
            range_collection.insert(insert_pos, range);
        }
    }
    DWARFDebugAranges::RangeColl& range_collection;
};

//----------------------------------------------------------------------
// PrintRange
//----------------------------------------------------------------------
static void PrintRange(const DWARFDebugAranges::Range& range)
{
    // Cast the address values in case the address type is compiled as 32 bit
    printf("0x%8.8x: [0x%8.8llx - 0x%8.8llx)\n", range.offset, (long long)range.lo_pc, (long long)range.hi_pc);
}

//----------------------------------------------------------------------
// Extract
//----------------------------------------------------------------------
bool
DWARFDebugAranges::Extract(const DataExtractor &debug_aranges_data)
{
    if (debug_aranges_data.ValidOffset(0))
    {
        uint32_t offset = 0;

        typedef std::vector<DWARFDebugArangeSet>    SetCollection;
        typedef SetCollection::const_iterator       SetCollectionIter;
        SetCollection sets;

        DWARFDebugArangeSet set;
        Range range;
        while (set.Extract(debug_aranges_data, &offset))
            sets.push_back(set);

        uint32_t count = 0;

        for_each(sets.begin(), sets.end(), CountArangeDescriptors(count));

        if (count > 0)
        {
            m_aranges.reserve(count);
            AddArangeDescriptors range_adder(m_aranges);
            for_each(sets.begin(), sets.end(), range_adder);
        }

    //  puts("\n\nDWARFDebugAranges list is:\n");
    //  for_each(m_aranges.begin(), m_aranges.end(), PrintRange);
    }
    return false;
}

//----------------------------------------------------------------------
// Generate
//----------------------------------------------------------------------
bool
DWARFDebugAranges::Generate(SymbolFileDWARF* dwarf2Data)
{
    Clear();
    DWARFDebugInfo* debug_info = dwarf2Data->DebugInfo();
    if (debug_info)
    {
        uint32_t cu_idx = 0;
        const uint32_t num_compile_units = dwarf2Data->GetNumCompileUnits();
        for (cu_idx = 0; cu_idx < num_compile_units; ++cu_idx)
        {
            DWARFCompileUnit* cu = debug_info->GetCompileUnitAtIndex(cu_idx);
            if (cu)
                cu->DIE()->BuildAddressRangeTable(dwarf2Data, cu, this);
        }
    }
    return !IsEmpty();
}


void
DWARFDebugAranges::Print() const
{
    puts("\n\nDWARFDebugAranges address range list is:\n");
    for_each(m_aranges.begin(), m_aranges.end(), PrintRange);
}


void
DWARFDebugAranges::Range::Dump(Stream *s) const
{
    s->Printf("{0x%8.8x}: [0x%8.8llx - 0x%8.8llx)\n", offset, lo_pc, hi_pc);
}

//----------------------------------------------------------------------
// Dump
//----------------------------------------------------------------------
//void
//DWARFDebugAranges::Dump(SymbolFileDWARF* dwarf2Data, Stream *s)
//{
//    const DataExtractor &debug_aranges_data = dwarf2Data->get_debug_aranges_data();
//    if (debug_aranges_data.ValidOffset(0))
//    {
//        uint32_t offset = 0;
//
//        DWARFDebugArangeSet set;
//        while (set.Extract(debug_aranges_data, &offset))
//            set.Dump(s);
//    }
//    else
//        s->PutCString("< EMPTY >\n");
//}
//

//----------------------------------------------------------------------
// AppendDebugRanges
//----------------------------------------------------------------------
//void
//DWARFDebugAranges::AppendDebugRanges(BinaryStreamBuf& debug_ranges, dw_addr_t cu_base_addr, uint32_t addr_size) const
//{
//  if (!m_aranges.empty())
//  {
//      RangeCollIterator end = m_aranges.end();
//      RangeCollIterator pos;
//      RangeCollIterator lo_pos = end;
//      for (pos = m_aranges.begin(); pos != end; ++pos)
//      {
//          if (lo_pos == end)
//              lo_pos = pos;
//
//          RangeCollIterator next = pos + 1;
//          if (next != end)
//          {
//              // Check to see if we can combine two consecutive ranges?
//              if (pos->hi_pc == next->lo_pc)
//                  continue;   // We can combine them!
//          }
//
//          if (cu_base_addr == 0 || cu_base_addr == DW_INVALID_ADDRESS)
//          {
//              debug_ranges.AppendMax64(lo_pos->lo_pc, addr_size);
//              debug_ranges.AppendMax64(pos->hi_pc, addr_size);
//          }
//          else
//          {
//              assert(lo_pos->lo_pc >= cu_base_addr);
//              assert(pos->hi_pc >= cu_base_addr);
//              debug_ranges.AppendMax64(lo_pos->lo_pc - cu_base_addr, addr_size);
//              debug_ranges.AppendMax64(pos->hi_pc - cu_base_addr, addr_size);
//          }
//
//          // Reset the low part of the next address range
//          lo_pos = end;
//      }
//  }
//  // Terminate the .debug_ranges with two zero addresses
//  debug_ranges.AppendMax64(0, addr_size);
//  debug_ranges.AppendMax64(0, addr_size);
//
//}
//
//----------------------------------------------------------------------
// ArangeSetContainsAddress
//----------------------------------------------------------------------
class ArangeSetContainsAddress
{
public:
    ArangeSetContainsAddress (dw_addr_t the_address) : address(the_address), offset(DW_INVALID_OFFSET) {}
    bool operator() (const DWARFDebugArangeSet& set)
    {
        offset = set.FindAddress(address);
        return (offset != DW_INVALID_OFFSET);
    }
    const dw_addr_t address;
    dw_offset_t offset;
};


//----------------------------------------------------------------------
// InsertRange
//----------------------------------------------------------------------
void
DWARFDebugAranges::InsertRange(dw_offset_t offset, dw_addr_t low_pc, dw_addr_t high_pc)
{
    // Insert each item in increasing address order so binary searching
    // can later be done!
    DWARFDebugAranges::Range range(low_pc, high_pc, offset);
    InsertRange(range);
}

//----------------------------------------------------------------------
// InsertRange
//----------------------------------------------------------------------
void
DWARFDebugAranges::InsertRange(const DWARFDebugAranges::Range& range)
{
    // Insert each item in increasing address order so binary searching
    // can later be done!
    RangeColl::iterator insert_pos = lower_bound(m_aranges.begin(), m_aranges.end(), range, RangeLessThan);
    m_aranges.insert(insert_pos, range);
}


void
DWARFDebugAranges::AppendRange (dw_offset_t offset, dw_addr_t low_pc, dw_addr_t high_pc)
{
    if (!m_aranges.empty())
    {
        if (m_aranges.back().offset == offset && m_aranges.back().hi_pc == low_pc)
        {
            m_aranges.back().hi_pc = high_pc;
            return;
        }
    }
    m_aranges.push_back (DWARFDebugAranges::Range(low_pc, high_pc, offset));
}

void
DWARFDebugAranges::Sort()
{    
    std::vector<size_t> indices;
    size_t end;
    Timer scoped_timer(__PRETTY_FUNCTION__, "%s this = %p",
                       __PRETTY_FUNCTION__, this);

    // Sort our address range entries
    std::stable_sort (m_aranges.begin(), m_aranges.end(), RangeLessThan);

    // Merge all neighbouring ranges into a single range and remember the
    // indices of all ranges merged.
    end = m_aranges.size();
    for (size_t merge, cursor = 1; cursor < end; ++cursor)
    {
        merge = cursor - 1;
        Range &r1 = m_aranges[merge];
        Range &r2 = m_aranges[cursor];

        if (r1.hi_pc == r2.lo_pc && r1.offset == r2.offset)
        {
            r2.lo_pc = r1.lo_pc;
            indices.push_back(merge);
        }
    }

    if (indices.empty())
        return;

    // Remove the merged ranges by shifting down all the keepers...
    std::set<size_t> purged(indices.begin(), indices.end());
    size_t new_size = m_aranges.size() - indices.size();
    for (size_t src = 0, dst = 0; dst < new_size; ++src, ++dst)
    {
        while (purged.count(src) > 0)
            ++src;
        if (src == dst)
            continue;
        m_aranges[dst] = m_aranges[src];
    }

    // ...and drop the extra elements.
    m_aranges.resize(new_size);
}

//----------------------------------------------------------------------
// FindAddress
//----------------------------------------------------------------------
dw_offset_t
DWARFDebugAranges::FindAddress(dw_addr_t address) const
{
    if ( !m_aranges.empty() )
    {
        DWARFDebugAranges::Range range(address);
        DWARFDebugAranges::RangeCollIterator begin = m_aranges.begin();
        DWARFDebugAranges::RangeCollIterator end = m_aranges.end();
        DWARFDebugAranges::RangeCollIterator pos = lower_bound(begin, end, range, RangeLessThan);

        if ((pos != end) && (pos->lo_pc <= address && address < pos->hi_pc))
        {
        //  printf("FindAddress(1) found 0x%8.8x in compile unit: 0x%8.8x\n", address, pos->offset);
            return pos->offset;
        }
        else if (pos != begin)
        {
            --pos;
            if ((pos->lo_pc <= address) && (address < pos->hi_pc))
            {
            //  printf("FindAddress(2) found 0x%8.8x in compile unit: 0x%8.8x\n", address, pos->offset);
                return (*pos).offset;
            }
        }
    }
    return DW_INVALID_OFFSET;
}

//----------------------------------------------------------------------
// AllRangesAreContiguous
//----------------------------------------------------------------------
bool
DWARFDebugAranges::AllRangesAreContiguous(dw_addr_t& lo_pc, dw_addr_t& hi_pc) const
{
    if (m_aranges.empty())
        return false;

    DWARFDebugAranges::RangeCollIterator begin = m_aranges.begin();
    DWARFDebugAranges::RangeCollIterator end = m_aranges.end();
    DWARFDebugAranges::RangeCollIterator pos;
    dw_addr_t next_addr = 0;

    for (pos = begin; pos != end; ++pos)
    {
        if ((pos != begin) && (pos->lo_pc != next_addr))
            return false;
        next_addr = pos->hi_pc;
    }
    lo_pc = m_aranges.front().lo_pc;    // We checked for empty at the start of function so front() will be valid
    hi_pc = m_aranges.back().hi_pc;     // We checked for empty at the start of function so back() will be valid
    return true;
}

bool
DWARFDebugAranges::GetMaxRange(dw_addr_t& lo_pc, dw_addr_t& hi_pc) const
{
    if (m_aranges.empty())
        return false;

    lo_pc = m_aranges.front().lo_pc;    // We checked for empty at the start of function so front() will be valid
    hi_pc = m_aranges.back().hi_pc;     // We checked for empty at the start of function so back() will be valid
    return true;
}

