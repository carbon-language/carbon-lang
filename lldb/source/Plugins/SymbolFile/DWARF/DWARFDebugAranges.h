//===-- DWARFDebugAranges.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DWARFDebugAranges_h_
#define SymbolFileDWARF_DWARFDebugAranges_h_

#include "DWARFDebugArangeSet.h"
#include <list>

class SymbolFileDWARF;

class DWARFDebugAranges
{
public:
    struct Range
    {
        Range(
            dw_addr_t   _lo_pc = DW_INVALID_ADDRESS,
            dw_addr_t   _hi_pc = DW_INVALID_ADDRESS,
            dw_offset_t _offset = DW_INVALID_OFFSET) :
                lo_pc(_lo_pc),
                hi_pc(_hi_pc),
                offset(_offset)
        {
        }

        void Clear()
        {
            lo_pc = hi_pc = DW_INVALID_ADDRESS;
            offset = DW_INVALID_OFFSET;
        }

        bool ValidRange() const
        {
            return hi_pc > lo_pc;
        }

        bool Contains(const Range& range) const
        {
            return lo_pc <= range.lo_pc && range.hi_pc <= hi_pc;
        }

        void Dump(lldb_private::Stream *s) const;
        dw_addr_t   lo_pc;      // Start of address range
        dw_addr_t   hi_pc;      // End of address range (not including this address)
        dw_offset_t offset;     // Offset of the compile unit or die
    };

                DWARFDebugAranges();

    void        Clear() { m_aranges.clear(); }
    bool        AllRangesAreContiguous(dw_addr_t& lo_pc, dw_addr_t& hi_pc) const;
    bool        GetMaxRange(dw_addr_t& lo_pc, dw_addr_t& hi_pc) const;
    bool        Extract(const lldb_private::DataExtractor &debug_aranges_data);
    bool        Generate(SymbolFileDWARF* dwarf2Data);
    void        InsertRange (dw_offset_t cu_offset, dw_addr_t low_pc, dw_addr_t high_pc);
    void        InsertRange (const DWARFDebugAranges::Range& range);
    
                // Use append range multiple times and then call sort
    void        AppendRange (dw_offset_t cu_offset, dw_addr_t low_pc, dw_addr_t high_pc);
    void        Sort();

    const Range* RangeAtIndex(uint32_t idx) const
                {
                    if (idx < m_aranges.size())
                        return &m_aranges[idx];
                    return NULL;
                }
    void        Print() const;
    dw_offset_t FindAddress(dw_addr_t address) const;
    bool        IsEmpty() const { return m_aranges.empty(); }
//    void        Dump(lldb_private::Stream *s);
    uint32_t    NumRanges() const
                {
                    return m_aranges.size();
                }

    dw_offset_t OffsetAtIndex(uint32_t idx) const
                {
                    if (idx < m_aranges.size())
                        return m_aranges[idx].offset;
                    return DW_INVALID_OFFSET;
                }
//  void    AppendDebugRanges(BinaryStreamBuf& debug_ranges, dw_addr_t cu_base_addr, uint32_t addr_size) const;

    static void Dump(SymbolFileDWARF* dwarf2Data, lldb_private::Stream *s);

    typedef std::vector<Range>              RangeColl;
    typedef RangeColl::const_iterator       RangeCollIterator;

protected:

    RangeColl m_aranges;
};


#endif  // SymbolFileDWARF_DWARFDebugAranges_h_
