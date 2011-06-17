//===-- DWARFDebugInfo.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolFileDWARF.h"

#include <algorithm>
#include <set>

#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"

#include "DWARFDebugInfo.h"
#include "DWARFCompileUnit.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFFormValue.h"

using namespace lldb_private;
using namespace std;

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
DWARFDebugInfo::DWARFDebugInfo() :
    m_dwarf2Data(NULL),
    m_compile_units()
{
}

//----------------------------------------------------------------------
// SetDwarfData
//----------------------------------------------------------------------
void
DWARFDebugInfo::SetDwarfData(SymbolFileDWARF* dwarf2Data)
{
    m_dwarf2Data = dwarf2Data;
    m_compile_units.clear();
}

//----------------------------------------------------------------------
// BuildDIEAddressRangeTable
//----------------------------------------------------------------------
bool
DWARFDebugInfo::BuildFunctionAddressRangeTable(DWARFDebugAranges* debug_aranges)
{
    const uint32_t num_compile_units = GetNumCompileUnits();
    uint32_t idx;
    for (idx = 0; idx < num_compile_units; ++idx)
    {
        DWARFCompileUnit* cu = GetCompileUnitAtIndex (idx);
        if (cu)
        {
            cu->DIE()->BuildFunctionAddressRangeTable(m_dwarf2Data, cu, debug_aranges);
        }
    }
    return !debug_aranges->IsEmpty();
}

//----------------------------------------------------------------------
// LookupAddress
//----------------------------------------------------------------------
bool
DWARFDebugInfo::LookupAddress
(
    const dw_addr_t address,
    const dw_offset_t hint_die_offset,
    DWARFCompileUnitSP& cu_sp,
    DWARFDebugInfoEntry** function_die,
    DWARFDebugInfoEntry** block_die
)
{

    if (hint_die_offset != DW_INVALID_OFFSET)
        cu_sp = GetCompileUnit(hint_die_offset);
    else
    {
        // Get a non const version of the address ranges
        DWARFDebugAranges* debug_aranges = ((SymbolFileDWARF*)m_dwarf2Data)->DebugAranges();

        if (debug_aranges != NULL)
        {
            // If we have an empty address ranges section, lets build a sorted
            // table ourselves by going through all of the debug information so we
            // can do quick subsequent searches.

            if (debug_aranges->IsEmpty())
            {
                const uint32_t num_compile_units = GetNumCompileUnits();
                uint32_t idx;
                for (idx = 0; idx < num_compile_units; ++idx)
                {
                    DWARFCompileUnit* cu = GetCompileUnitAtIndex(idx);
                    if (cu)
                        cu->DIE()->BuildAddressRangeTable(m_dwarf2Data, cu, debug_aranges);
                }
            }
            cu_sp = GetCompileUnit(debug_aranges->FindAddress(address));
        }
    }

    if (cu_sp.get())
    {
        if (cu_sp->LookupAddress(address, function_die, block_die))
            return true;
        cu_sp.reset();
    }
    else
    {
        // The hint_die_offset may have been a pointer to the actual item that
        // we are looking for
        DWARFDebugInfoEntry* die_ptr = GetDIEPtr(hint_die_offset, &cu_sp);
        if (die_ptr)
        {
            if (cu_sp.get())
            {
                if (function_die || block_die)
                    return die_ptr->LookupAddress(address, m_dwarf2Data, cu_sp.get(), function_die, block_die);

                // We only wanted the compile unit that contained this address
                return true;
            }
        }
    }
    return false;
}


void
DWARFDebugInfo::ParseCompileUnitHeadersIfNeeded()
{
    if (m_compile_units.empty())
    {
        if (m_dwarf2Data != NULL)
        {
            uint32_t offset = 0;
            const DataExtractor &debug_info_data = m_dwarf2Data->get_debug_info_data();
            while (debug_info_data.ValidOffset(offset))
            {
                DWARFCompileUnitSP cu_sp(new DWARFCompileUnit(m_dwarf2Data));
                // Out of memory?
                if (cu_sp.get() == NULL)
                    break;

                if (cu_sp->Extract(debug_info_data, &offset) == false)
                    break;

                m_compile_units.push_back(cu_sp);

                offset = cu_sp->GetNextCompileUnitOffset();
            }
        }
    }
}

uint32_t
DWARFDebugInfo::GetNumCompileUnits()
{
    ParseCompileUnitHeadersIfNeeded();
    return m_compile_units.size();
}

DWARFCompileUnit*
DWARFDebugInfo::GetCompileUnitAtIndex(uint32_t idx)
{
    DWARFCompileUnit* cu = NULL;
    if (idx < GetNumCompileUnits())
        cu = m_compile_units[idx].get();
    return cu;
}

static bool CompileUnitOffsetLessThan (const DWARFCompileUnitSP& a, const DWARFCompileUnitSP& b)
{
    return a->GetOffset() < b->GetOffset();
}


static int
CompareDWARFCompileUnitSPOffset (const void *key, const void *arrmem)
{
    const dw_offset_t key_cu_offset = *(dw_offset_t*) key;
    const dw_offset_t cu_offset = ((DWARFCompileUnitSP *)arrmem)->get()->GetOffset();
    if (key_cu_offset < cu_offset)
        return -1;
    if (key_cu_offset > cu_offset)
        return 1;
    return 0;
}

DWARFCompileUnitSP
DWARFDebugInfo::GetCompileUnit(dw_offset_t cu_offset, uint32_t* idx_ptr)
{
    DWARFCompileUnitSP cu_sp;
    uint32_t cu_idx = DW_INVALID_INDEX;
    if (cu_offset != DW_INVALID_OFFSET)
    {
        ParseCompileUnitHeadersIfNeeded();

        DWARFCompileUnitSP* match = (DWARFCompileUnitSP*)bsearch(&cu_offset, &m_compile_units[0], m_compile_units.size(), sizeof(DWARFCompileUnitSP), CompareDWARFCompileUnitSPOffset);
        if (match)
        {
            cu_sp = *match;
            cu_idx = match - &m_compile_units[0];
        }
    }
    if (idx_ptr)
        *idx_ptr = cu_idx;
    return cu_sp;
}

DWARFCompileUnitSP
DWARFDebugInfo::GetCompileUnitContainingDIE(dw_offset_t die_offset)
{
    DWARFCompileUnitSP cu_sp;
    if (die_offset != DW_INVALID_OFFSET)
    {
        ParseCompileUnitHeadersIfNeeded();

        CompileUnitColl::const_iterator end_pos = m_compile_units.end();
        CompileUnitColl::const_iterator pos;

        for (pos = m_compile_units.begin(); pos != end_pos; ++pos)
        {
            dw_offset_t cu_start_offset = (*pos)->GetOffset();
            dw_offset_t cu_end_offset = (*pos)->GetNextCompileUnitOffset();
            if (cu_start_offset <= die_offset && die_offset < cu_end_offset)
            {
                cu_sp = *pos;
                break;
            }
        }
    }
    return cu_sp;
}

//----------------------------------------------------------------------
// Compare function DWARFDebugAranges::Range structures
//----------------------------------------------------------------------
static bool CompareDIEOffset (const DWARFDebugInfoEntry& die1, const DWARFDebugInfoEntry& die2)
{
    return die1.GetOffset() < die2.GetOffset();
}


//----------------------------------------------------------------------
// GetDIE()
//
// Get the DIE (Debug Information Entry) with the specified offset.
//----------------------------------------------------------------------
DWARFDebugInfoEntry*
DWARFDebugInfo::GetDIEPtr(dw_offset_t die_offset, DWARFCompileUnitSP* cu_sp_ptr)
{
    DWARFCompileUnitSP cu_sp(GetCompileUnitContainingDIE(die_offset));
    if (cu_sp_ptr)
        *cu_sp_ptr = cu_sp;
    if (cu_sp.get())
        return cu_sp->GetDIEPtr(die_offset);
    return NULL;    // Not found in any compile units
}

DWARFDebugInfoEntry*
DWARFDebugInfo::GetDIEPtrWithCompileUnitHint (dw_offset_t die_offset, DWARFCompileUnit**cu_handle)
{
    assert (cu_handle);
    DWARFDebugInfoEntry* die = NULL;
    if (*cu_handle)
        die = (*cu_handle)->GetDIEPtr(die_offset);

    if (die == NULL)
    {
        DWARFCompileUnitSP cu_sp (GetCompileUnitContainingDIE(die_offset));
        if (cu_sp.get())
        {
            *cu_handle = cu_sp.get();
            die = cu_sp->GetDIEPtr(die_offset);
        }
    }
    if (die == NULL)
        *cu_handle = NULL;
    return die;
}


const DWARFDebugInfoEntry*
DWARFDebugInfo::GetDIEPtrContainingOffset(dw_offset_t die_offset, DWARFCompileUnitSP* cu_sp_ptr)
{
    DWARFCompileUnitSP cu_sp(GetCompileUnitContainingDIE(die_offset));
    if (cu_sp_ptr)
        *cu_sp_ptr = cu_sp;
    if (cu_sp.get())
        return cu_sp->GetDIEPtrContainingOffset(die_offset);

    return NULL;    // Not found in any compile units

}

//----------------------------------------------------------------------
// DWARFDebugInfo_ParseCallback
//
// A callback function for the static DWARFDebugInfo::Parse() function
// that gets parses all compile units and DIE's into an internate
// representation for further modification.
//----------------------------------------------------------------------

static dw_offset_t
DWARFDebugInfo_ParseCallback
(
    SymbolFileDWARF* dwarf2Data,
    DWARFCompileUnitSP& cu_sp,
    DWARFDebugInfoEntry* die,
    const dw_offset_t next_offset,
    const uint32_t curr_depth,
    void* userData
)
{
    DWARFDebugInfo* debug_info = (DWARFDebugInfo*)userData;
    DWARFCompileUnit* cu = cu_sp.get();
    if (die)
    {
        cu->AddDIE(*die);
    }
    else if (cu)
    {
        debug_info->AddCompileUnit(cu_sp);
    }

    // Just return the current offset to parse the next CU or DIE entry
    return next_offset;
}

//----------------------------------------------------------------------
// AddCompileUnit
//----------------------------------------------------------------------
void
DWARFDebugInfo::AddCompileUnit(DWARFCompileUnitSP& cu)
{
    m_compile_units.push_back(cu);
}

/*
void
DWARFDebugInfo::AddDIE(DWARFDebugInfoEntry& die)
{
    m_die_array.push_back(die);
}
*/




//----------------------------------------------------------------------
// Parse
//
// Parses the .debug_info section and uses the .debug_abbrev section
// and various other sections in the SymbolFileDWARF class and calls the
// supplied callback function each time a compile unit header, or debug
// information entry is successfully parsed. This function can be used
// for different tasks such as parsing the file contents into a
// structured data, dumping, verifying and much more.
//----------------------------------------------------------------------
void
DWARFDebugInfo::Parse(SymbolFileDWARF* dwarf2Data, Callback callback, void* userData)
{
    if (dwarf2Data)
    {
        uint32_t offset = 0;
        uint32_t depth = 0;
        DWARFCompileUnitSP cu(new DWARFCompileUnit(dwarf2Data));
        if (cu.get() == NULL)
            return;
        DWARFDebugInfoEntry die;

        while (cu->Extract(dwarf2Data->get_debug_info_data(), &offset))
        {
            const dw_offset_t next_cu_offset = cu->GetNextCompileUnitOffset();

            depth = 0;
            // Call the callback function with no DIE pointer for the compile unit
            // and get the offset that we are to continue to parse from
            offset = callback(dwarf2Data, cu, NULL, offset, depth, userData);

            // Make sure we are within our compile unit
            if (offset < next_cu_offset)
            {
                // We are in our compile unit, parse starting at the offset
                // we were told to parse
                bool done = false;
                while (!done && die.Extract(dwarf2Data, cu.get(), &offset))
                {
                    // Call the callback function with DIE pointer that falls within the compile unit
                    offset = callback(dwarf2Data, cu, &die, offset, depth, userData);

                    if (die.IsNULL())
                    {
                        if (depth)
                            --depth;
                        else
                            done = true;    // We are done with this compile unit!
                    }
                    else if (die.HasChildren())
                        ++depth;
                }
            }

            // Make sure the offset returned is valid, and if not stop parsing.
            // Returning DW_INVALID_OFFSET from this callback is a good way to end
            // all parsing
            if (!dwarf2Data->get_debug_info_data().ValidOffset(offset))
                break;

            // See if during the callback anyone retained a copy of the compile
            // unit other than ourselves and if so, let whomever did own the object
            // and create a new one for our own use!
            if (!cu.unique())
                cu.reset(new DWARFCompileUnit(dwarf2Data));


            // Make sure we start on a proper
            offset = next_cu_offset;
        }
    }
}

/*
typedef struct AddressRangeTag
{
    dw_addr_t   lo_pc;
    dw_addr_t   hi_pc;
    dw_offset_t die_offset;
} AddressRange;
*/
struct DIERange
{
    DIERange() :
        range(),
        lo_die_offset(),
        hi_die_offset()
    {
    }

    DWARFDebugAranges::Range range;
    dw_offset_t lo_die_offset;
    dw_offset_t hi_die_offset;
};

typedef struct DwarfStat
{
    DwarfStat() : count(0), byte_size(0) {}
    uint32_t count;
    uint32_t byte_size;
} DwarfStat;

typedef map<dw_attr_t, DwarfStat> DwarfAttrStatMap;

typedef struct DIEStat
{
    DIEStat() : count(0), byte_size(0), attr_stats() {}
    uint32_t count;
    uint32_t byte_size;
    DwarfAttrStatMap attr_stats;
} DIEStat;

typedef map<dw_tag_t, DIEStat> DIEStatMap;
struct VerifyInfo
{
    VerifyInfo(Stream* the_strm) :
        strm(the_strm),
        die_ranges(),
        addr_range_errors(0),
        sibling_errors(0),
        die_stats()
    {
    }

    Stream* strm;
    vector<DIERange> die_ranges;
    uint32_t addr_range_errors;
    uint32_t sibling_errors;
    DIEStatMap die_stats;

    DISALLOW_COPY_AND_ASSIGN(VerifyInfo);

};


//----------------------------------------------------------------------
// VerifyCallback
//
// A callback function for the static DWARFDebugInfo::Parse() function
// that gets called each time a compile unit header or debug information
// entry is successfully parsed.
//
// This function will verify the DWARF information is well formed by
// making sure that any DW_TAG_compile_unit tags that have valid address
// ranges (DW_AT_low_pc and DW_AT_high_pc) have no gaps in the address
// ranges of it contained DW_TAG_subprogram tags. Also the sibling chain
// and relationships are verified to make sure nothing gets hosed up
// when dead stripping occurs.
//----------------------------------------------------------------------

static dw_offset_t
VerifyCallback
(
    SymbolFileDWARF* dwarf2Data,
    DWARFCompileUnitSP& cu_sp,
    DWARFDebugInfoEntry* die,
    const dw_offset_t next_offset,
    const uint32_t curr_depth,
    void* userData
)
{
    VerifyInfo* verifyInfo = (VerifyInfo*)userData;

    const DWARFCompileUnit* cu = cu_sp.get();
    Stream *s = verifyInfo->strm;
    bool verbose = s->GetVerbose();
    if (die)
    {
    //  die->Dump(dwarf2Data, cu, f);
        const DWARFAbbreviationDeclaration* abbrevDecl = die->GetAbbreviationDeclarationPtr();
        // We have a DIE entry
        if (abbrevDecl)
        {
            const dw_offset_t die_offset = die->GetOffset();
            const dw_offset_t sibling = die->GetAttributeValueAsReference(dwarf2Data, cu, DW_AT_sibling, DW_INVALID_OFFSET);

            if (sibling != DW_INVALID_OFFSET)
            {
                if (sibling <= next_offset)
                {
                    if (verifyInfo->sibling_errors++ == 0)
                        s->Printf("ERROR\n");
                    s->Printf("    0x%8.8x: sibling attribute (0x%8.8x) in this die is not valid: it is less than this DIE or some of its contents.\n", die->GetOffset(), sibling);
                }
                else if (sibling > verifyInfo->die_ranges.back().hi_die_offset)
                {
                    if (verifyInfo->sibling_errors++ == 0)
                        s->Printf("ERROR\n");
                    s->Printf("    0x%8.8x: sibling attribute (0x%8.8x) in this DIE is not valid: it is greater than the end of the parent scope.\n", die->GetOffset(), sibling);
                }
            }

            if ((die_offset < verifyInfo->die_ranges.back().lo_die_offset) || (die_offset >= verifyInfo->die_ranges.back().hi_die_offset))
            {
                if (verifyInfo->sibling_errors++ == 0)
                    s->Printf("ERROR\n");
                s->Printf("    0x%8.8x: DIE offset is not within the parent DIE range {0x%8.8x}: (0x%8.8x - 0x%8.8x)\n",
                        die->GetOffset(),
                        verifyInfo->die_ranges.back().range.offset,
                        verifyInfo->die_ranges.back().lo_die_offset,
                        verifyInfo->die_ranges.back().hi_die_offset);

            }

            dw_tag_t tag = abbrevDecl->Tag();

            // Keep some stats on this DWARF file
            verifyInfo->die_stats[tag].count++;
            verifyInfo->die_stats[tag].byte_size += (next_offset - die->GetOffset());

            if (verbose)
            {
                DIEStat& tag_stat = verifyInfo->die_stats[tag];

                const DataExtractor& debug_info = dwarf2Data->get_debug_info_data();

                dw_offset_t offset = die->GetOffset();
                // Skip the abbreviation code so we are at the data for the attributes
                debug_info.Skip_LEB128(&offset);

                const uint32_t numAttributes = abbrevDecl->NumAttributes();
                dw_attr_t attr;
                dw_form_t form;
                for (uint32_t idx = 0; idx < numAttributes; ++idx)
                {
                    dw_offset_t start_offset = offset;
                    abbrevDecl->GetAttrAndFormByIndexUnchecked(idx, attr, form);
                    DWARFFormValue::SkipValue(form, debug_info, &offset, cu);

                    if (tag_stat.attr_stats.find(attr) == tag_stat.attr_stats.end())
                    {
                        tag_stat.attr_stats[attr].count = 0;
                        tag_stat.attr_stats[attr].byte_size = 0;
                    }

                    tag_stat.attr_stats[attr].count++;
                    tag_stat.attr_stats[attr].byte_size += offset - start_offset;
                }
            }

            DWARFDebugAranges::Range range;
            range.offset = die->GetOffset();

            switch (tag)
            {
            case DW_TAG_compile_unit:
                // Check for previous subroutines that were within a previous
                //
            //  VerifyAddressRangesForCU(verifyInfo);
                // Remember which compile unit we are dealing with so we can verify
                // the address ranges within it (if any) are contiguous. The DWARF
                // spec states that if a compile unit TAG has high and low PC
                // attributes, there must be no gaps in the address ranges of it's
                // contained subroutines. If there are gaps, the high and low PC
                // must not be in the DW_TAG_compile_unit's attributes. Errors like
                // this can crop up when optimized code is dead stripped and the debug
                // information isn't properly fixed up for output.
                range.lo_pc = die->GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_low_pc, DW_INVALID_ADDRESS);
                if (range.lo_pc != DW_INVALID_ADDRESS)
                {
                    range.hi_pc = die->GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_high_pc, DW_INVALID_ADDRESS);
                    if (s->GetVerbose())
                    {
                        s->Printf("\n    CU ");
                        range.Dump(s);
                    }
                }
                else
                {
                    range.lo_pc = die->GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_entry_pc, DW_INVALID_ADDRESS);
                }
                break;

            case DW_TAG_subprogram:
                // If the DW_TAG_compile_unit that contained this function had a
                // valid address range, add all of the valid subroutine address
                // ranges to a collection of addresses which will be sorted
                // and verified right before the next DW_TAG_compile_unit is
                // processed to make sure that there are no gaps in the address
                // range.
                range.lo_pc = die->GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_low_pc, DW_INVALID_ADDRESS);
                if (range.lo_pc != DW_INVALID_ADDRESS)
                {
                    range.hi_pc = die->GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_high_pc, DW_INVALID_ADDRESS);
                    if (range.hi_pc != DW_INVALID_ADDRESS)
                    {
                        range.offset = die->GetOffset();
                        bool valid = range.ValidRange();
                        if (!valid || s->GetVerbose())
                        {
                            s->Printf("\n  FUNC ");
                            range.Dump(s);
                            if (!valid)
                            {
                                ++verifyInfo->addr_range_errors;
                                s->Printf(" ERROR: Invalid address range for function.");
                            }
                        }

                        // Only add to our subroutine ranges if our compile unit has a valid address range
                    //  if (valid && verifyInfo->die_ranges.size() >= 2 && verifyInfo->die_ranges[1].range.ValidRange())
                    //      verifyInfo->subroutine_ranges.InsertRange(range);
                    }
                }
                break;

            case DW_TAG_lexical_block:
            case DW_TAG_inlined_subroutine:
                {
                    range.lo_pc = die->GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_low_pc, DW_INVALID_ADDRESS);
                    if (range.lo_pc != DW_INVALID_ADDRESS)
                    {
                        range.hi_pc = die->GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_high_pc, DW_INVALID_ADDRESS);
                        if (range.hi_pc != DW_INVALID_ADDRESS)
                        {
                            range.offset = die->GetOffset();
                            bool valid = range.ValidRange();
                            if (!valid || s->GetVerbose())
                            {
                                s->Printf("\n  BLCK ");
                                range.Dump(s);
                                if (!valid)
                                {
                                    ++verifyInfo->addr_range_errors;
                                    s->Printf(" ERROR: Invalid address range for block or inlined subroutine.");
                                }
                            }
                        }
                    }
                }
                break;
            }

            if (range.ValidRange() && verifyInfo->die_ranges.back().range.ValidRange())
            {
                if (!verifyInfo->die_ranges.back().range.Contains(range))
                {
                    ++verifyInfo->addr_range_errors;
                    s->Printf("\n       ");
                    range.Dump(s);
                    s->Printf(" ERROR: Range is not in parent");
                    verifyInfo->die_ranges.back().range.Dump(s);
                }
            }

            if (die->HasChildren())
            {
                // Keep tabs on the valid address ranges for the current item to make
                // sure that it all fits (make sure the sibling offsets got fixed up
                // correctly if any functions were dead stripped).
                DIERange die_range;
                die_range.range = range;
                die_range.lo_die_offset = next_offset;
                die_range.hi_die_offset = sibling;
                if (die_range.hi_die_offset == DW_INVALID_OFFSET)
                    die_range.hi_die_offset = verifyInfo->die_ranges.back().hi_die_offset;
                verifyInfo->die_ranges.push_back(die_range);
            }
        }
        else
        {
            // NULL entry
            verifyInfo->die_ranges.pop_back();
        }
    }
    else
    {
    //  cu->Dump(ostrm_ptr); // Dump the compile unit for the DIE
        // We have a new compile unit header
        verifyInfo->die_ranges.clear();
        DIERange die_range;
        die_range.range.offset  = cu->GetOffset();
        die_range.lo_die_offset = next_offset;
        die_range.hi_die_offset = cu->GetNextCompileUnitOffset();
        verifyInfo->die_ranges.push_back(die_range);
    }

    // Just return the current offset to parse the next CU or DIE entry
    return next_offset;
}


class CompareDIEStatSizes
{
public:
    bool operator() (const DIEStatMap::const_iterator& pos1, const DIEStatMap::const_iterator& pos2) const
    {
        return pos1->second.byte_size <= pos2->second.byte_size;
    }
};

class CompareAttrDIEStatSizes
{
public:
    bool operator() (const DwarfAttrStatMap::const_iterator& pos1, const DwarfAttrStatMap::const_iterator& pos2) const
    {
        return pos1->second.byte_size <= pos2->second.byte_size;
    }
};

//----------------------------------------------------------------------
// Verify
//
// Verifies the DWARF information is valid.
//----------------------------------------------------------------------
void
DWARFDebugInfo::Verify(Stream *s, SymbolFileDWARF* dwarf2Data)
{
    s->Printf("Verifying Compile Unit Header chain.....");
    VerifyInfo verifyInfo(s);
    verifyInfo.addr_range_errors = 0;
    verifyInfo.sibling_errors = 0;

    bool verbose = s->GetVerbose();

    uint32_t offset = 0;
    if (verbose)
        s->EOL();
//  vector<dw_offset_t> valid_cu_offsets;
    DWARFCompileUnit cu (dwarf2Data);
    bool success = true;
    while ( success && dwarf2Data->get_debug_info_data().ValidOffset(offset+cu.Size()) )
    {
        success = cu.Extract (dwarf2Data->get_debug_info_data(), &offset);
        if (!success)
            s->Printf("ERROR\n");
    //  else
    //      valid_cu_offsets.push_back(cu.GetOffset());

        cu.Verify(verifyInfo.strm);
        offset = cu.GetNextCompileUnitOffset();
    }

    if (success)
        s->Printf("OK\n");

    s->Printf("Verifying address ranges and siblings...");
    if (verbose)
        s->EOL();
    DWARFDebugInfo::Parse(dwarf2Data, VerifyCallback, &verifyInfo);

//  VerifyAddressRangesForCU(&verifyInfo);

    if (verifyInfo.addr_range_errors > 0)
        s->Printf("\nERRORS - %u error(s) were found.\n", verifyInfo.addr_range_errors);
    else
        s->Printf("OK\n");

    uint32_t total_category_sizes[kNumTagCategories] = {0};
    uint32_t total_category_count[kNumTagCategories] = {0};
    uint32_t total_die_count = 0;
    uint32_t total_die_size = 0;

    typedef set<DIEStatMap::const_iterator, CompareDIEStatSizes> DIEStatBySizeMap;

    s->PutCString(  "\n"
                "DWARF Statistics\n"
                "Count    Size     Size %   Tag\n"
                "-------- -------- -------- -------------------------------------------\n");
    DIEStatBySizeMap statBySizeMap;
    DIEStatMap::const_iterator pos;
    DIEStatMap::const_iterator end_pos = verifyInfo.die_stats.end();
    for (pos = verifyInfo.die_stats.begin(); pos != end_pos; ++pos)
    {
        const uint32_t die_count = pos->second.count;
        const uint32_t die_size = pos->second.byte_size;

        statBySizeMap.insert(pos);
        total_die_count += die_count;
        total_die_size += die_size;
        DW_TAG_CategoryEnum category = get_tag_category(pos->first);
        total_category_sizes[category] += die_size;
        total_category_count[category] += die_count;
    }

    float total_die_size_float = total_die_size;

    DIEStatBySizeMap::const_reverse_iterator size_pos;
    DIEStatBySizeMap::const_reverse_iterator size_pos_end = statBySizeMap.rend();
    float percentage;
    for (size_pos = statBySizeMap.rbegin(); size_pos != size_pos_end; ++size_pos)
    {
        pos = *size_pos;

        const DIEStat& tag_stat = pos->second;

        const uint32_t die_count = tag_stat.count;
        const uint32_t die_size = tag_stat.byte_size;
        percentage = ((float)die_size/total_die_size_float)*100.0;
        s->Printf("%7u %8u %2.2f%%     %s\n", die_count, die_size, percentage, DW_TAG_value_to_name(pos->first));

        const DwarfAttrStatMap& attr_stats = tag_stat.attr_stats;
        if (!attr_stats.empty())
        {
            typedef set<DwarfAttrStatMap::const_iterator, CompareAttrDIEStatSizes> DwarfAttrStatBySizeMap;
            DwarfAttrStatBySizeMap attrStatBySizeMap;
            DwarfAttrStatMap::const_iterator attr_stat_pos;
            DwarfAttrStatMap::const_iterator attr_stat_pos_end = attr_stats.end();
            for (attr_stat_pos = attr_stats.begin(); attr_stat_pos != attr_stat_pos_end; ++attr_stat_pos)
            {
                attrStatBySizeMap.insert(attr_stat_pos);
            }

            DwarfAttrStatBySizeMap::const_reverse_iterator attr_size_pos;
            DwarfAttrStatBySizeMap::const_reverse_iterator attr_size_pos_end = attrStatBySizeMap.rend();
            for (attr_size_pos = attrStatBySizeMap.rbegin(); attr_size_pos != attr_size_pos_end; ++attr_size_pos)
            {
                attr_stat_pos = *attr_size_pos;
                percentage = ((float)attr_stat_pos->second.byte_size/die_size)*100.0;
                s->Printf("%7u %8u %2.2f%%    %s\n", attr_stat_pos->second.count, attr_stat_pos->second.byte_size, percentage, DW_AT_value_to_name(attr_stat_pos->first));
            }
            s->EOL();
        }
    }

    s->Printf("-------- -------- -------- -------------------------------------------\n");
    s->Printf("%7u %8u 100.00% Total for all DIEs\n", total_die_count, total_die_size);

    float total_category_percentages[kNumTagCategories] =
    {
        ((float)total_category_sizes[TagCategoryVariable]/total_die_size_float)*100.0,
        ((float)total_category_sizes[TagCategoryType]/total_die_size_float)*100.0,
        ((float)total_category_sizes[TagCategoryProgram]/total_die_size_float)*100.0
    };

    s->EOL();
    s->Printf("%7u %8u %2.2f%%    %s\n", total_category_count[TagCategoryVariable], total_category_sizes[TagCategoryVariable],  total_category_percentages[TagCategoryVariable],    "Total for variable related DIEs");
    s->Printf("%7u %8u %2.2f%%    %s\n", total_category_count[TagCategoryType],     total_category_sizes[TagCategoryType],      total_category_percentages[TagCategoryType],        "Total for type related DIEs");
    s->Printf("%7u %8u %2.2f%%    %s\n", total_category_count[TagCategoryProgram],      total_category_sizes[TagCategoryProgram],   total_category_percentages[TagCategoryProgram],     "Total for program related DIEs");
    s->Printf("\n\n");
}

typedef struct DumpInfo
{
    DumpInfo(Stream* init_strm, uint32_t off, uint32_t depth) :
        strm(init_strm),
        die_offset(off),
        recurse_depth(depth),
        found_depth(UINT32_MAX),
        found_die(false),
        ancestors()
    {
    }
    Stream* strm;
    const uint32_t die_offset;
    const uint32_t recurse_depth;
    uint32_t found_depth;
    bool found_die;
    std::vector<DWARFDebugInfoEntry> ancestors;

    DISALLOW_COPY_AND_ASSIGN(DumpInfo);
} DumpInfo;

//----------------------------------------------------------------------
// DumpCallback
//
// A callback function for the static DWARFDebugInfo::Parse() function
// that gets called each time a compile unit header or debug information
// entry is successfully parsed.
//
// This function dump DWARF information and obey recurse depth and
// whether a single DIE is to be dumped (or all of the data).
//----------------------------------------------------------------------
static dw_offset_t DumpCallback
(
    SymbolFileDWARF* dwarf2Data,
    DWARFCompileUnitSP& cu_sp,
    DWARFDebugInfoEntry* die,
    const dw_offset_t next_offset,
    const uint32_t curr_depth,
    void* userData
)
{
    DumpInfo* dumpInfo = (DumpInfo*)userData;

    const DWARFCompileUnit* cu = cu_sp.get();

    Stream *s = dumpInfo->strm;
    bool show_parents = s->GetFlags().Test(DWARFDebugInfo::eDumpFlag_ShowAncestors);

    if (die)
    {
        // Are we dumping everything?
        if (dumpInfo->die_offset == DW_INVALID_OFFSET)
        {
            // Yes we are dumping everything. Obey our recurse level though
            if (curr_depth < dumpInfo->recurse_depth)
                die->Dump(dwarf2Data, cu, s, 0);
        }
        else
        {
            // We are dumping a specific DIE entry by offset
            if (dumpInfo->die_offset == die->GetOffset())
            {
                // We found the DIE we were looking for, dump it!
                if (show_parents)
                {
                    s->SetIndentLevel(0);
                    const uint32_t num_ancestors = dumpInfo->ancestors.size();
                    if (num_ancestors > 0)
                    {
                        for (uint32_t i=0; i<num_ancestors-1; ++i)
                        {
                            dumpInfo->ancestors[i].Dump(dwarf2Data, cu, s, 0);
                            s->IndentMore();
                        }
                    }
                }

                dumpInfo->found_depth = curr_depth;

                die->Dump(dwarf2Data, cu, s, 0);

                // Note that we found the DIE we were looking for
                dumpInfo->found_die = true;

                // Since we are dumping a single DIE, if there are no children we are done!
                if (!die->HasChildren() || dumpInfo->recurse_depth == 0)
                    return DW_INVALID_OFFSET;   // Return an invalid address to end parsing
            }
            else if (dumpInfo->found_die)
            {
                // Are we done with all the children?
                if (curr_depth <= dumpInfo->found_depth)
                    return DW_INVALID_OFFSET;

                // We have already found our DIE and are printing it's children. Obey
                // our recurse depth and return an invalid offset if we get done
                // dumping all the the children
                if (dumpInfo->recurse_depth == UINT32_MAX || curr_depth <= dumpInfo->found_depth + dumpInfo->recurse_depth)
                    die->Dump(dwarf2Data, cu, s, 0);
            }
            else if (dumpInfo->die_offset > die->GetOffset())
            {
                if (show_parents)
                    dumpInfo->ancestors.back() = *die;
            }
        }

        // Keep up with our indent level
        if (die->IsNULL())
        {
            if (show_parents)
                dumpInfo->ancestors.pop_back();

            if (curr_depth <= 1)
                return cu->GetNextCompileUnitOffset();
            else
                s->IndentLess();
        }
        else if (die->HasChildren())
        {
            if (show_parents)
            {
                DWARFDebugInfoEntry null_die;
                dumpInfo->ancestors.push_back(null_die);
            }
            s->IndentMore();
        }
    }
    else
    {
        if (cu == NULL)
            s->PutCString("NULL - cu");
        // We have a compile unit, reset our indent level to zero just in case
        s->SetIndentLevel(0);

        // See if we are dumping everything?
        if (dumpInfo->die_offset == DW_INVALID_OFFSET)
        {
            // We are dumping everything
            cu->Dump(s);
            return cu->GetFirstDIEOffset(); // Return true to parse all DIEs in this Compile Unit
        }
        else
        {
            if (show_parents)
            {
                dumpInfo->ancestors.clear();
                dumpInfo->ancestors.resize(1);
            }

            // We are dumping only a single DIE possibly with it's children and
            // we must find it's compile unit before we can dump it properly
            if (dumpInfo->die_offset < cu->GetFirstDIEOffset())
            {
                // Not found, maybe the DIE offset provided wasn't correct?
            //  *ostrm_ptr << "DIE at offset " << HEX32 << dumpInfo->die_offset << " was not found." << endl;
                return DW_INVALID_OFFSET;
            }
            else
            {
                // See if the DIE is in this compile unit?
                if (dumpInfo->die_offset < cu->GetNextCompileUnitOffset())
                {
                    // This DIE is in this compile unit!
                    if (s->GetVerbose())
                        cu->Dump(s); // Dump the compile unit for the DIE in verbose mode

                    return next_offset;
                //  // We found our compile unit that contains our DIE, just skip to dumping the requested DIE...
                //  return dumpInfo->die_offset;
                }
                else
                {
                    // Skip to the next compile unit as the DIE isn't in the current one!
                    return cu->GetNextCompileUnitOffset();
                }
            }
        }
    }

    // Just return the current offset to parse the next CU or DIE entry
    return next_offset;
}

//----------------------------------------------------------------------
// Dump
//
// Dump the information in the .debug_info section to the specified
// ostream. If die_offset is valid, a single DIE will be dumped. If the
// die_offset is invalid, all the DWARF information will be dumped. Both
// cases will obey a "recurse_depth" or how deep to traverse into the
// children of each DIE entry. A recurse_depth of zero will dump all
// compile unit headers. A recurse_depth of 1 will dump all compile unit
// headers and the DW_TAG_compile unit tags. A depth of 2 will also
// dump all types and functions.
//----------------------------------------------------------------------
void
DWARFDebugInfo::Dump
(
    Stream *s,
    SymbolFileDWARF* dwarf2Data,
    const uint32_t die_offset,
    const uint32_t recurse_depth
)
{
    DumpInfo dumpInfo(s, die_offset, recurse_depth);
    s->PutCString(".debug_info contents");
    if (dwarf2Data->get_debug_info_data().GetByteSize() > 0)
    {
        if (die_offset == DW_INVALID_OFFSET)
            s->PutCString(":\n");
        else
        {
            s->Printf(" for DIE entry at .debug_info[0x%8.8x]", die_offset);
            if (recurse_depth != UINT32_MAX)
                s->Printf(" recursing %u levels deep.", recurse_depth);
            s->EOL();
        }
    }
    else
    {
        s->PutCString(": < EMPTY >\n");
        return;
    }
    DWARFDebugInfo::Parse(dwarf2Data, DumpCallback, &dumpInfo);
}


//----------------------------------------------------------------------
// Dump
//
// Dump the contents of this DWARFDebugInfo object as has been parsed
// and/or modified after it has been parsed.
//----------------------------------------------------------------------
void
DWARFDebugInfo::Dump (Stream *s, const uint32_t die_offset, const uint32_t recurse_depth)
{
    DumpInfo dumpInfo(s, die_offset, recurse_depth);

    s->PutCString("Dumping .debug_info section from internal representation\n");

    CompileUnitColl::const_iterator pos;
    uint32_t curr_depth = 0;
    ParseCompileUnitHeadersIfNeeded();
    for (pos = m_compile_units.begin(); pos != m_compile_units.end(); ++pos)
    {
        const DWARFCompileUnitSP& cu_sp = *pos;
        DumpCallback(m_dwarf2Data, (DWARFCompileUnitSP&)cu_sp, NULL, 0, curr_depth, &dumpInfo);
        cu_sp->DIE()->Dump(m_dwarf2Data, cu_sp.get(), s, recurse_depth);
    }
}


//----------------------------------------------------------------------
// FindCallbackString
//
// A callback function for the static DWARFDebugInfo::Parse() function
// that gets called each time a compile unit header or debug information
// entry is successfully parsed.
//
// This function will find the die_offset of any items whose DW_AT_name
// matches the given string
//----------------------------------------------------------------------
typedef struct FindCallbackStringInfoTag
{
    const char* name;
    bool ignore_case;
    RegularExpression* regex;
    vector<dw_offset_t>& die_offsets;
} FindCallbackStringInfo;

static dw_offset_t FindCallbackString
(
    SymbolFileDWARF* dwarf2Data,
    DWARFCompileUnitSP& cu_sp,
    DWARFDebugInfoEntry* die,
    const dw_offset_t next_offset,
    const uint32_t curr_depth,
    void* userData
)
{
    FindCallbackStringInfo* info = (FindCallbackStringInfo*)userData;
    const DWARFCompileUnit* cu = cu_sp.get();

    if (die)
    {
        const char* die_name = die->GetName(dwarf2Data, cu);
        if (die_name)
        {
            if (info->regex)
            {
                if (info->regex->Execute(die_name))
                    info->die_offsets.push_back(die->GetOffset());
            }
            else
            {
                if ((info->ignore_case ? strcasecmp(die_name, info->name) : strcmp(die_name, info->name)) == 0)
                    info->die_offsets.push_back(die->GetOffset());
            }
        }
    }

    // Just return the current offset to parse the next CU or DIE entry
    return next_offset;
}

//----------------------------------------------------------------------
// Find
//
// Finds all DIE that have a specific DW_AT_name attribute by manually
// searching through the debug information (not using the
// .debug_pubnames section). The string must match the entire name
// and case sensitive searches are an option.
//----------------------------------------------------------------------
bool
DWARFDebugInfo::Find(const char* name, bool ignore_case, vector<dw_offset_t>& die_offsets) const
{
    die_offsets.clear();
    if (name && name[0])
    {
        FindCallbackStringInfo info = { name, ignore_case, NULL, die_offsets };
        DWARFDebugInfo::Parse(m_dwarf2Data, FindCallbackString, &info);
    }
    return !die_offsets.empty();
}

//----------------------------------------------------------------------
// Find
//
// Finds all DIE that have a specific DW_AT_name attribute by manually
// searching through the debug information (not using the
// .debug_pubnames section). The string must match the supplied regular
// expression.
//----------------------------------------------------------------------
bool
DWARFDebugInfo::Find(RegularExpression& re, vector<dw_offset_t>& die_offsets) const
{
    die_offsets.clear();
    FindCallbackStringInfo info = { NULL, false, &re, die_offsets };
    DWARFDebugInfo::Parse(m_dwarf2Data, FindCallbackString, &info);
    return !die_offsets.empty();
}
