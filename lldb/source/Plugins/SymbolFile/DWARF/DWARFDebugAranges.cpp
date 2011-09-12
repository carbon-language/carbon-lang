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

#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"

#include "LogChannelDWARF.h"
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
            range.length = arange_desc_ptr->length;

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
    printf("0x%8.8x: [0x%8.8llx - 0x%8.8llx)\n", range.offset, (long long)range.lo_pc, (long long)range.hi_pc());
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
        const bool clear_dies_if_already_not_parsed = true;
        uint32_t cu_idx = 0;
        const uint32_t num_compile_units = dwarf2Data->GetNumCompileUnits();
        for (cu_idx = 0; cu_idx < num_compile_units; ++cu_idx)
        {
            DWARFCompileUnit* cu = debug_info->GetCompileUnitAtIndex(cu_idx);
            if (cu)
                cu->BuildAddressRangeTable(dwarf2Data, this, clear_dies_if_already_not_parsed);
        }
    }
    return !IsEmpty();
}


void
DWARFDebugAranges::Dump (Log *log) const
{
    if (log == NULL)
        return;
    const uint32_t num_ranges = NumRanges();
    for (uint32_t i = 0; i < num_ranges; ++i)
    {
        const Range &range = m_aranges[i];
        log->Printf ("0x%8.8x: [0x%8.8llx - 0x%8.8llx)", range.offset, (uint64_t)range.lo_pc, (uint64_t)range.hi_pc());
    }
}


void
DWARFDebugAranges::Range::Dump(Stream *s) const
{
    s->Printf("{0x%8.8x}: [0x%8.8llx - 0x%8.8llx)\n", offset, lo_pc, hi_pc());
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
void
DWARFDebugAranges::AppendRange (dw_offset_t offset, dw_addr_t low_pc, dw_addr_t high_pc)
{
    if (!m_aranges.empty())
    {
        if (m_aranges.back().offset == offset && m_aranges.back().hi_pc() == low_pc)
        {
            m_aranges.back().set_hi_pc(high_pc);
            return;
        }
    }
    m_aranges.push_back (DWARFDebugAranges::Range(low_pc, high_pc, offset));
}

void
DWARFDebugAranges::Sort (bool minimize, uint32_t n)
{    
    Timer scoped_timer(__PRETTY_FUNCTION__, "%s this = %p",
                       __PRETTY_FUNCTION__, this);

    Log *log = LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_ARANGES);
    const size_t orig_arange_size = m_aranges.size();
    if (log)
    {
        log->Printf ("DWARFDebugAranges::Sort(minimize = %u, n = %u) with %zu entries", minimize, n, orig_arange_size);
        Dump (log);
    }

    // Size of one? If so, no sorting is needed
    if (orig_arange_size <= 1)
        return;
    // Sort our address range entries
    std::stable_sort (m_aranges.begin(), m_aranges.end(), RangeLessThan);

    
    if (!minimize)
        return;

    // Most address ranges are contiguous from function to function
    // so our new ranges will likely be smaller. We calculate the size
    // of the new ranges since although std::vector objects can be resized, 
    // the will never reduce their allocated block size and free any excesss
    // memory, so we might as well start a brand new collection so it is as
    // small as possible.

    // First calculate the size of the new minimal arange vector
    // so we don't have to do a bunch of re-allocations as we 
    // copy the new minimal stuff over to the new collection
    size_t minimal_size = 1;
    size_t i;
    for (i=1; i<orig_arange_size; ++i)
    {
        if (!DWARFDebugAranges::Range::SortedOverlapCheck (m_aranges[i-1], m_aranges[i], n))
            ++minimal_size;
    }

    // If the sizes are the same, then no consecutive aranges can be 
    // combined, we are done
    if (minimal_size == orig_arange_size)
        return;

    // Else, make a new RangeColl that _only_ contains what we need.
    RangeColl minimal_aranges;
    minimal_aranges.resize(minimal_size);
    uint32_t j=0;
    minimal_aranges[j] = m_aranges[0];
    for (i=1; i<orig_arange_size; ++i)
    {
        if (DWARFDebugAranges::Range::SortedOverlapCheck (minimal_aranges[j], m_aranges[i], n))
        {
            minimal_aranges[j].set_hi_pc (m_aranges[i].hi_pc());
        }
        else
        {
            // Only increment j if we aren't merging
            minimal_aranges[++j] = m_aranges[i];            
        }
    }
    assert (j+1 == minimal_size);
    
    // Now swap our new minimal aranges into place. The local
    // minimal_aranges will then contian the old big collection
    // which will get freed.
    minimal_aranges.swap(m_aranges);
    
    if (log)
    {
        size_t delta = orig_arange_size - m_aranges.size();
        log->Printf ("DWARFDebugAranges::Sort() %zu entries after minimizing (%zu entries combined for %zu bytes saved)", 
                     m_aranges.size(), delta, delta * sizeof(Range));
        Dump (log);
    }
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

        if ((pos != end) && (pos->lo_pc <= address && address < pos->hi_pc()))
        {
        //  printf("FindAddress(1) found 0x%8.8x in compile unit: 0x%8.8x\n", address, pos->offset);
            return pos->offset;
        }
        else if (pos != begin)
        {
            --pos;
            if ((pos->lo_pc <= address) && (address < pos->hi_pc()))
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
        next_addr = pos->hi_pc();
    }
    lo_pc = m_aranges.front().lo_pc;    // We checked for empty at the start of function so front() will be valid
    hi_pc = m_aranges.back().hi_pc();     // We checked for empty at the start of function so back() will be valid
    return true;
}

bool
DWARFDebugAranges::GetMaxRange(dw_addr_t& lo_pc, dw_addr_t& hi_pc) const
{
    if (m_aranges.empty())
        return false;

    lo_pc = m_aranges.front().lo_pc;    // We checked for empty at the start of function so front() will be valid
    hi_pc = m_aranges.back().hi_pc();   // We checked for empty at the start of function so back() will be valid
    return true;
}

