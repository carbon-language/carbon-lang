//===-- DWARFDebugRanges.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DWARFDebugRanges_h_
#define liblldb_DWARFDebugRanges_h_

#include "SymbolFileDWARF.h"
#include <map>
#include <vector>


class DWARFDebugRanges
{
public:

    //------------------------------------------------------------------
    // Address range
    //------------------------------------------------------------------
    struct Range
    {
        Range(dw_addr_t begin = DW_INVALID_ADDRESS, dw_addr_t end = DW_INVALID_ADDRESS) :
            begin_offset(begin),
            end_offset(end)
        {
        }

        void Clear()
        {
            begin_offset = DW_INVALID_ADDRESS;
            end_offset = DW_INVALID_ADDRESS;
        }

        dw_addr_t   begin_offset;
        dw_addr_t   end_offset;

        typedef std::vector<Range>          collection;
        typedef collection::iterator        iterator;
        typedef collection::const_iterator  const_iterator;

    };

    //------------------------------------------------------------------
    // Collection of ranges
    //------------------------------------------------------------------
    struct RangeList
    {
            RangeList() :
                ranges()
            {
            }

        bool Extract(SymbolFileDWARF* dwarf2Data, uint32_t* offset_ptr);
        bool AddRange(dw_addr_t lo_addr, dw_addr_t hi_addr);
        void Clear()
            {
                ranges.clear();
            }

        dw_addr_t LowestAddress(const dw_addr_t base_addr) const;
        dw_addr_t HighestAddress(const dw_addr_t base_addr) const;
        void AddOffset(dw_addr_t offset);
        void SubtractOffset(dw_addr_t offset);
        size_t Size() const;
        const Range* RangeAtIndex(size_t i) const;
        const Range* Lookup(dw_addr_t offset) const;
        Range::collection   ranges;
    };

    DWARFDebugRanges();
    ~DWARFDebugRanges();
    void Extract(SymbolFileDWARF* dwarf2Data);
    static void Dump(lldb_private::Stream *s, const lldb_private::DataExtractor& debug_ranges_data, uint32_t* offset_ptr, dw_addr_t cu_base_addr);
    bool FindRanges(dw_offset_t debug_ranges_offset, DWARFDebugRanges::RangeList& range_list) const;

protected:
    typedef std::map<dw_offset_t, RangeList>    range_map;
    typedef range_map::iterator                 range_map_iterator;
    typedef range_map::const_iterator           range_map_const_iterator;
    range_map m_range_map;
};


#endif  // liblldb_DWARFDebugRanges_h_
