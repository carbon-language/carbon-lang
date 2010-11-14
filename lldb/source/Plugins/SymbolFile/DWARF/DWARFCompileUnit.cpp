//===-- DWARFCompileUnit.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFCompileUnit.h"

#include "lldb/Core/Mangled.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"

#include "DWARFDebugAbbrev.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFDIECollection.h"
#include "DWARFFormValue.h"
#include "LogChannelDWARF.h"
#include "NameToDIE.h"
#include "SymbolFileDWARF.h"

using namespace lldb_private;
using namespace std;

extern int g_verbose;

DWARFCompileUnit::DWARFCompileUnit(SymbolFileDWARF* dwarf2Data) :
    m_dwarf2Data    ( dwarf2Data ),
    m_offset        ( DW_INVALID_OFFSET ),
    m_length        ( 0 ),
    m_version       ( 0 ),
    m_abbrevs       ( NULL ),
    m_addr_size     ( DWARFCompileUnit::GetDefaultAddressSize() ),
    m_base_addr     ( 0 ),
    m_die_array     (),
    m_aranges_ap    (),
    m_user_data     ( NULL )
{
}

void
DWARFCompileUnit::Clear()
{
    m_offset        = DW_INVALID_OFFSET;
    m_length        = 0;
    m_version       = 0;
    m_abbrevs       = NULL;
    m_addr_size     = DWARFCompileUnit::GetDefaultAddressSize();
    m_base_addr     = 0;
    m_die_array.clear();
    m_aranges_ap.reset();
    m_user_data     = NULL;
}

bool
DWARFCompileUnit::Extract(const DataExtractor &debug_info, uint32_t* offset_ptr)
{
    Clear();

    m_offset = *offset_ptr;

    if (debug_info.ValidOffset(*offset_ptr))
    {
        dw_offset_t abbr_offset;
        const DWARFDebugAbbrev *abbr = m_dwarf2Data->DebugAbbrev();
        m_length        = debug_info.GetU32(offset_ptr);
        m_version       = debug_info.GetU16(offset_ptr);
        abbr_offset     = debug_info.GetU32(offset_ptr);
        m_addr_size     = debug_info.GetU8 (offset_ptr);

        bool length_OK = debug_info.ValidOffset(GetNextCompileUnitOffset()-1);
        bool version_OK = SymbolFileDWARF::SupportedVersion(m_version);
        bool abbr_offset_OK = m_dwarf2Data->get_debug_abbrev_data().ValidOffset(abbr_offset);
        bool addr_size_OK = ((m_addr_size == 4) || (m_addr_size == 8));

        if (length_OK && version_OK && addr_size_OK && abbr_offset_OK && abbr != NULL)
        {
            m_abbrevs = abbr->GetAbbreviationDeclarationSet(abbr_offset);
            return true;
        }

        // reset the offset to where we tried to parse from if anything went wrong
        *offset_ptr = m_offset;
    }

    return false;
}


dw_offset_t
DWARFCompileUnit::Extract(dw_offset_t offset, const DataExtractor& debug_info_data, const DWARFAbbreviationDeclarationSet* abbrevs)
{
    Clear();

    m_offset = offset;

    if (debug_info_data.ValidOffset(offset))
    {
        m_length        = debug_info_data.GetU32(&offset);
        m_version       = debug_info_data.GetU16(&offset);
        bool abbrevs_OK = debug_info_data.GetU32(&offset) == abbrevs->GetOffset();
        m_abbrevs       = abbrevs;
        m_addr_size     = debug_info_data.GetU8 (&offset);

        bool version_OK = SymbolFileDWARF::SupportedVersion(m_version);
        bool addr_size_OK = ((m_addr_size == 4) || (m_addr_size == 8));

        if (version_OK && addr_size_OK && abbrevs_OK && debug_info_data.ValidOffset(offset))
            return offset;
    }
    return DW_INVALID_OFFSET;
}

void
DWARFCompileUnit::ClearDIEs(bool keep_compile_unit_die)
{
    if (m_die_array.size() > 1)
    {
        // std::vectors never get any smaller when resized to a smaller size,
        // or when clear() or erase() are called, the size will report that it
        // is smaller, but the memory allocated remains intact (call capacity()
        // to see this). So we need to create a temporary vector and swap the
        // contents which will cause just the internal pointers to be swapped
        // so that when "tmp_array" goes out of scope, it will destroy the
        // contents.

        // Save at least the compile unit DIE
        DWARFDebugInfoEntry::collection tmp_array;
        m_die_array.swap(tmp_array);
        if (keep_compile_unit_die)
            m_die_array.push_back(tmp_array.front());
    }
}

//----------------------------------------------------------------------
// ParseCompileUnitDIEsIfNeeded
//
// Parses a compile unit and indexes its DIEs if it hasn't already been
// done.
//----------------------------------------------------------------------
size_t
DWARFCompileUnit::ExtractDIEsIfNeeded (bool cu_die_only)
{
    const size_t initial_die_array_size = m_die_array.size();
    if ((cu_die_only && initial_die_array_size > 0) || initial_die_array_size > 1)
        return 0; // Already parsed

    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "%8.8x: DWARFCompileUnit::ExtractDIEsIfNeeded( cu_die_only = %i )",
                        m_offset,
                        cu_die_only);

    // Set the offset to that of the first DIE
    uint32_t offset = GetFirstDIEOffset();
    DWARFDebugInfoEntry die;
        // Keep a flat array of the DIE for binary lookup by DIE offset
//    Log *log = LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO);
//        if (log)
//            log->Printf("0x%8.8x: Compile Unit: length = 0x%8.8x, version = 0x%4.4x, abbr_offset = 0x%8.8x, addr_size = 0x%2.2x",
//                        cu->GetOffset(),
//                        cu->GetLength(),
//                        cu->GetVersion(),
//                        cu->GetAbbrevOffset(),
//                        cu->GetAddressByteSize());

    uint32_t depth = 0;
    // We are in our compile unit, parse starting at the offset
    // we were told to parse
    const DataExtractor& debug_info_data = m_dwarf2Data->get_debug_info_data();

    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (GetAddressByteSize());
    while (die.FastExtract (debug_info_data, this, fixed_form_sizes, &offset))
    {
//        if (log)
//            log->Printf("0x%8.8x: %*.*s%s%s",
//                        die.GetOffset(),
//                        depth * 2, depth * 2, "",
//                        DW_TAG_value_to_name (die.Tag()),
//                        die.HasChildren() ? " *" : "");

        if (depth == 0)
        {
            uint64_t base_addr = die.GetAttributeValueAsUnsigned(m_dwarf2Data, this, DW_AT_low_pc, LLDB_INVALID_ADDRESS);
            if (base_addr == LLDB_INVALID_ADDRESS)
                base_addr = die.GetAttributeValueAsUnsigned(m_dwarf2Data, this, DW_AT_entry_pc, 0);
            SetBaseAddress (base_addr);
        }

        if (cu_die_only)
        {
            AddDIE (die);
            return 1;
        }
        else if (depth == 0 && initial_die_array_size == 1)
        {
            // Don't append the CU die as we already did that
        }
        else
        {
            AddDIE (die);
        }

        const DWARFAbbreviationDeclaration* abbrDecl = die.GetAbbreviationDeclarationPtr();
        if (abbrDecl)
        {
            // Normal DIE
            if (abbrDecl->HasChildren())
                ++depth;
        }
        else
        {
            // NULL DIE.
            if (depth > 0)
                --depth;
            if (depth == 0)
                break;  // We are done with this compile unit!
        }

        assert(offset <= GetNextCompileUnitOffset());
    }
    SetDIERelations();
    return m_die_array.size();
}


dw_offset_t
DWARFCompileUnit::GetAbbrevOffset() const
{
    return m_abbrevs ? m_abbrevs->GetOffset() : DW_INVALID_OFFSET;
}



bool
DWARFCompileUnit::Verify(Stream *s) const
{
    const DataExtractor& debug_info = m_dwarf2Data->get_debug_info_data();
    bool valid_offset = debug_info.ValidOffset(m_offset);
    bool length_OK = debug_info.ValidOffset(GetNextCompileUnitOffset()-1);
    bool version_OK = SymbolFileDWARF::SupportedVersion(m_version);
    bool abbr_offset_OK = m_dwarf2Data->get_debug_abbrev_data().ValidOffset(GetAbbrevOffset());
    bool addr_size_OK = ((m_addr_size == 4) || (m_addr_size == 8));
    bool verbose = s->GetVerbose();
    if (valid_offset && length_OK && version_OK && addr_size_OK && abbr_offset_OK)
    {
        if (verbose)
            s->Printf("    0x%8.8x: OK\n", m_offset);
        return true;
    }
    else
    {
        s->Printf("    0x%8.8x: ", m_offset);

        m_dwarf2Data->get_debug_info_data().Dump (s, m_offset, lldb::eFormatHex, 1, Size(), 32, LLDB_INVALID_ADDRESS, 0, 0);
        s->EOL();
        if (valid_offset)
        {
            if (!length_OK)
                s->Printf("        The length (0x%8.8x) for this compile unit is too large for the .debug_info provided.\n", m_length);
            if (!version_OK)
                s->Printf("        The 16 bit compile unit header version is not supported.\n");
            if (!abbr_offset_OK)
                s->Printf("        The offset into the .debug_abbrev section (0x%8.8x) is not valid.\n", GetAbbrevOffset());
            if (!addr_size_OK)
                s->Printf("        The address size is unsupported: 0x%2.2x\n", m_addr_size);
        }
        else
            s->Printf("        The start offset of the compile unit header in the .debug_info is invalid.\n");
    }
    return false;
}


void
DWARFCompileUnit::Dump(Stream *s) const
{
    s->Printf("0x%8.8x: Compile Unit: length = 0x%8.8x, version = 0x%4.4x, abbr_offset = 0x%8.8x, addr_size = 0x%2.2x (next CU at {0x%8.8x})\n",
                m_offset, m_length, m_version, GetAbbrevOffset(), m_addr_size, GetNextCompileUnitOffset());
}


static uint8_t g_default_addr_size = 4;

uint8_t
DWARFCompileUnit::GetAddressByteSize(const DWARFCompileUnit* cu)
{
    if (cu)
        return cu->GetAddressByteSize();
    return DWARFCompileUnit::GetDefaultAddressSize();
}

uint8_t
DWARFCompileUnit::GetDefaultAddressSize()
{
    return g_default_addr_size;
}

void
DWARFCompileUnit::SetDefaultAddressSize(uint8_t addr_size)
{
    g_default_addr_size = addr_size;
}

bool
DWARFCompileUnit::LookupAddress
(
    const dw_addr_t address,
    DWARFDebugInfoEntry** function_die_handle,
    DWARFDebugInfoEntry** block_die_handle
)
{
    bool success = false;

    if (function_die_handle != NULL && DIE())
    {
        if (m_aranges_ap.get() == NULL)
        {
            m_aranges_ap.reset(new DWARFDebugAranges());
            m_die_array.front().BuildFunctionAddressRangeTable(m_dwarf2Data, this, m_aranges_ap.get());
        }

        // Re-check the aranges auto pointer contents in case it was created above
        if (m_aranges_ap.get() != NULL)
        {
            *function_die_handle = GetDIEPtr(m_aranges_ap->FindAddress(address));
            if (*function_die_handle != NULL)
            {
                success = true;
                if (block_die_handle != NULL)
                {
                    DWARFDebugInfoEntry* child = (*function_die_handle)->GetFirstChild();
                    while (child)
                    {
                        if (child->LookupAddress(address, m_dwarf2Data, this, NULL, block_die_handle))
                            break;
                        child = child->GetSibling();
                    }
                }
            }
        }
    }
    return success;
}

//----------------------------------------------------------------------
// SetDIERelations()
//
// We read in all of the DIE entries into our flat list of DIE entries
// and now we need to go back through all of them and set the parent,
// sibling and child pointers for quick DIE navigation.
//----------------------------------------------------------------------
void
DWARFCompileUnit::SetDIERelations()
{
#if 0
    // Compute average bytes per DIE
    //
    // We can figure out what the average number of bytes per DIE is
    // to help us pre-allocate the correct number of m_die_array
    // entries so we don't end up doing a lot of memory copies as we
    // are creating our DIE array when parsing
    //
    // Enable this code by changing "#if 0" above to "#if 1" and running
    // the dsymutil or dwarfdump with a bunch of dwarf files and see what
    // the running average ends up being in the stdout log.
    static size_t g_total_cu_debug_info_size = 0;
    static size_t g_total_num_dies = 0;
    static size_t g_min_bytes_per_die = UINT32_MAX;
    static size_t g_max_bytes_per_die = 0;
    const size_t num_dies = m_die_array.size();
    const size_t cu_debug_info_size = GetDebugInfoSize();
    const size_t bytes_per_die = cu_debug_info_size / num_dies;
    if (g_min_bytes_per_die > bytes_per_die)
        g_min_bytes_per_die = bytes_per_die;
    if (g_max_bytes_per_die < bytes_per_die)
        g_max_bytes_per_die = bytes_per_die;
    if (g_total_cu_debug_info_size == 0)
    {
        cout << "                    min max avg" << endl
             << "n dies cu size  bpd bpd bpd bpd" << endl
             << "------ -------- --- === === ===" << endl;
    }
    g_total_cu_debug_info_size += cu_debug_info_size;
    g_total_num_dies += num_dies;
    const size_t avg_bytes_per_die = g_total_cu_debug_info_size / g_total_num_dies;
    cout
        << DECIMAL_WIDTH(6) << num_dies << ' '
        << DECIMAL_WIDTH(8) << cu_debug_info_size  << ' '
        << DECIMAL_WIDTH(3) << bytes_per_die << ' '
        << DECIMAL_WIDTH(3) << g_min_bytes_per_die << ' '
        << DECIMAL_WIDTH(3) << g_max_bytes_per_die << ' '
        << DECIMAL_WIDTH(3) << avg_bytes_per_die
        << endl;
#endif
    if (m_die_array.empty())
        return;
    DWARFDebugInfoEntry* die_array_begin = &m_die_array.front();
    DWARFDebugInfoEntry* die_array_end = &m_die_array.back();
    DWARFDebugInfoEntry* curr_die;
    // We purposely are skipping the last element in the array in the loop below
    // so that we can always have a valid next item
    for (curr_die = die_array_begin;  curr_die < die_array_end;  ++curr_die)
    {
        // Since our loop doesn't include the last element, we can always
        // safely access the next die in the array.
        DWARFDebugInfoEntry* next_die = curr_die + 1;

        const DWARFAbbreviationDeclaration* curr_die_abbrev = curr_die->GetAbbreviationDeclarationPtr();

        if (curr_die_abbrev)
        {
            // Normal DIE
            if (curr_die_abbrev->HasChildren())
                next_die->SetParent(curr_die);
            else
                curr_die->SetSibling(next_die);
        }
        else
        {
            // NULL DIE that terminates a sibling chain
            DWARFDebugInfoEntry* parent = curr_die->GetParent();
            if (parent)
                parent->SetSibling(next_die);
        }
    }

    // Since we skipped the last element, we need to fix it up!
    if (die_array_begin < die_array_end)
        curr_die->SetParent(die_array_begin);

#if 0
    // The code below will dump the DIE relations in case any modification
    // is done to the above code. This dump can be used in a diff to make
    // sure that no functionality is lost.
    {
        DWARFDebugInfoEntry::const_iterator pos;
        DWARFDebugInfoEntry::const_iterator end = m_die_array.end();
        puts("offset    parent   sibling  child");
        puts("--------  -------- -------- --------");
        for (pos = m_die_array.begin(); pos != end; ++pos)
        {
            const DWARFDebugInfoEntry& die_ref = *pos;
            const DWARFDebugInfoEntry* p = die_ref.GetParent();
            const DWARFDebugInfoEntry* s = die_ref.GetSibling();
            const DWARFDebugInfoEntry* c = die_ref.GetFirstChild();
            printf("%.8x: %.8x %.8x %.8x\n", die_ref.GetOffset(),
                p ? p->GetOffset() : 0,
                s ? s->GetOffset() : 0,
                c ? c->GetOffset() : 0);
        }
    }
#endif

}
//----------------------------------------------------------------------
// Compare function DWARFDebugAranges::Range structures
//----------------------------------------------------------------------
static bool CompareDIEOffset (const DWARFDebugInfoEntry& die1, const DWARFDebugInfoEntry& die2)
{
    return die1.GetOffset() < die2.GetOffset();
}

//----------------------------------------------------------------------
// GetDIEPtr()
//
// Get the DIE (Debug Information Entry) with the specified offset.
//----------------------------------------------------------------------
DWARFDebugInfoEntry*
DWARFCompileUnit::GetDIEPtr(dw_offset_t die_offset)
{
    if (die_offset != DW_INVALID_OFFSET)
    {
        ExtractDIEsIfNeeded (false);
        DWARFDebugInfoEntry compare_die;
        compare_die.SetOffset(die_offset);
        DWARFDebugInfoEntry::iterator end = m_die_array.end();
        DWARFDebugInfoEntry::iterator pos = lower_bound(m_die_array.begin(), end, compare_die, CompareDIEOffset);
        if (pos != end)
        {
            if (die_offset == (*pos).GetOffset())
                return &(*pos);
        }
    }
    return NULL;    // Not found in any compile units
}

//----------------------------------------------------------------------
// GetDIEPtrContainingOffset()
//
// Get the DIE (Debug Information Entry) that contains the specified
// .debug_info offset.
//----------------------------------------------------------------------
const DWARFDebugInfoEntry*
DWARFCompileUnit::GetDIEPtrContainingOffset(dw_offset_t die_offset)
{
    if (die_offset != DW_INVALID_OFFSET)
    {
        ExtractDIEsIfNeeded (false);
        DWARFDebugInfoEntry compare_die;
        compare_die.SetOffset(die_offset);
        DWARFDebugInfoEntry::iterator end = m_die_array.end();
        DWARFDebugInfoEntry::iterator pos = lower_bound(m_die_array.begin(), end, compare_die, CompareDIEOffset);
        if (pos != end)
        {
            if (die_offset >= (*pos).GetOffset())
            {
                DWARFDebugInfoEntry::iterator next = pos + 1;
                if (next != end)
                {
                    if (die_offset < (*next).GetOffset())
                        return &(*pos);
                }
            }
        }
    }
    return NULL;    // Not found in any compile units
}



size_t
DWARFCompileUnit::AppendDIEsWithTag (const dw_tag_t tag, DWARFDIECollection& dies, uint32_t depth) const
{
    size_t old_size = dies.Size();
    DWARFDebugInfoEntry::const_iterator pos;
    DWARFDebugInfoEntry::const_iterator end = m_die_array.end();
    for (pos = m_die_array.begin(); pos != end; ++pos)
    {
        if (pos->Tag() == tag)
            dies.Append (&(*pos));
    }

    // Return the number of DIEs added to the collection
    return dies.Size() - old_size;
}

//void
//DWARFCompileUnit::AddGlobalDIEByIndex (uint32_t die_idx)
//{
//    m_global_die_indexes.push_back (die_idx);
//}
//
//
//void
//DWARFCompileUnit::AddGlobal (const DWARFDebugInfoEntry* die)
//{
//    // Indexes to all file level global and static variables
//    m_global_die_indexes;
//    
//    if (m_die_array.empty())
//        return;
//    
//    const DWARFDebugInfoEntry* first_die = &m_die_array[0];
//    const DWARFDebugInfoEntry* end = first_die + m_die_array.size();
//    if (first_die <= die && die < end)
//        m_global_die_indexes.push_back (die - first_die);
//}


void
DWARFCompileUnit::Index 
(
    const uint32_t cu_idx,
    NameToDIE& func_basenames,
    NameToDIE& func_fullnames,
    NameToDIE& func_methods,
    NameToDIE& func_selectors,
    NameToDIE& objc_class_selectors,
    NameToDIE& globals,
    NameToDIE& types,
    NameToDIE& namespaces,
    const DWARFDebugRanges *debug_ranges,
    DWARFDebugAranges *aranges
)
{
    const DataExtractor* debug_str = &m_dwarf2Data->get_debug_str_data();

    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (GetAddressByteSize());

    NameToDIE::Info die_info = { cu_idx, 0 };
    DWARFDebugInfoEntry::const_iterator pos;
    DWARFDebugInfoEntry::const_iterator begin = m_die_array.begin();
    DWARFDebugInfoEntry::const_iterator end = m_die_array.end();
    for (pos = begin; pos != end; ++pos)
    {
        const DWARFDebugInfoEntry &die = *pos;
        
        const dw_tag_t tag = die.Tag();
    
        switch (tag)
        {
        case DW_TAG_subprogram:
        case DW_TAG_inlined_subroutine:
        case DW_TAG_base_type:
        case DW_TAG_class_type:
        case DW_TAG_constant:
        case DW_TAG_enumeration_type:
        case DW_TAG_string_type:
        case DW_TAG_subroutine_type:
        case DW_TAG_structure_type:
        case DW_TAG_union_type:
        case DW_TAG_typedef:
        case DW_TAG_namespace:
        case DW_TAG_variable:
            break;
            
        default:
            continue;
        }

        DWARFDebugInfoEntry::Attributes attributes;
        const char *name = NULL;
        Mangled mangled;
        bool is_variable = false;
        bool is_declaration = false;
        bool is_artificial = false;
        bool has_address = false;
        bool has_location = false;
        bool is_global_or_static_variable = false;
        dw_addr_t lo_pc = DW_INVALID_ADDRESS;
        dw_addr_t hi_pc = DW_INVALID_ADDRESS;
        DWARFDebugRanges::RangeList ranges;

        dw_offset_t specification_die_offset = DW_INVALID_OFFSET;
        const size_t num_attributes = die.GetAttributes(m_dwarf2Data, this, fixed_form_sizes, attributes);
        if (num_attributes > 0)
        {
            uint32_t i;
            
            is_variable = tag == DW_TAG_variable;

            for (i=0; i<num_attributes; ++i)
            {
                dw_attr_t attr = attributes.AttributeAtIndex(i);
                DWARFFormValue form_value;
                switch (attr)
                {
                case DW_AT_name:
                    if (attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                        name = form_value.AsCString(debug_str);
                    break;

                case DW_AT_declaration:
                    if (attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                        is_declaration = form_value.Unsigned() != 0;
                    break;

                case DW_AT_artificial:
                    if (attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                        is_artificial = form_value.Unsigned() != 0;
                    break;

                case DW_AT_MIPS_linkage_name:
                    if (attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                        mangled.GetMangledName().SetCString(form_value.AsCString(debug_str));
                    break;

                case DW_AT_low_pc:
                    has_address = true;
                    if (tag == DW_TAG_subprogram && attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                    {
                        lo_pc = form_value.Unsigned();
                    }
                    break;

                case DW_AT_high_pc:
                    has_address = true;
                    if (tag == DW_TAG_subprogram && attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                    {
                        hi_pc = form_value.Unsigned();
                    }
                    break;

                case DW_AT_ranges:
                    if (tag == DW_TAG_subprogram && attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                    {
                        if (debug_ranges)
                        {
                            debug_ranges->FindRanges(form_value.Unsigned(), ranges);
                            // All DW_AT_ranges are relative to the base address of the
                            // compile unit. We add the compile unit base address to make
                            // sure all the addresses are properly fixed up.
                            ranges.AddOffset(GetBaseAddress());
                        }
                    }
                    has_address = true;
                    break;

                case DW_AT_entry_pc:
                    has_address = true;
                    break;

                case DW_AT_location:
                    has_location = true;
                    if (tag == DW_TAG_variable)
                    {
                        const DWARFDebugInfoEntry* parent_die = die.GetParent();
                        while ( parent_die != NULL )
                        {
                            switch (parent_die->Tag())
                            {
                            case DW_TAG_subprogram:
                            case DW_TAG_lexical_block:
                            case DW_TAG_inlined_subroutine:
                                // Even if this is a function level static, we don't add it. We could theoretically
                                // add these if we wanted to by introspecting into the DW_AT_location and seeing
                                // if the location describes a hard coded address, but we dont want the performance
                                // penalty of that right now.
                                is_global_or_static_variable = false;
//                              if (attributes.ExtractFormValueAtIndex(dwarf2Data, i, form_value))
//                              {
//                                  // If we have valid block data, then we have location expression bytes
//                                  // that are fixed (not a location list).
//                                  const uint8_t *block_data = form_value.BlockData();
//                                  if (block_data)
//                                  {
//                                      uint32_t block_length = form_value.Unsigned();
//                                      if (block_length == 1 + attributes.CompileUnitAtIndex(i)->GetAddressByteSize())
//                                      {
//                                          if (block_data[0] == DW_OP_addr)
//                                              add_die = true;
//                                      }
//                                  }
//                              }
                                parent_die = NULL;  // Terminate the while loop.
                                break;

                            case DW_TAG_compile_unit:
                                is_global_or_static_variable = true;
                                parent_die = NULL;  // Terminate the while loop.
                                break;

                            default:
                                parent_die = parent_die->GetParent();   // Keep going in the while loop.
                                break;
                            }
                        }
                    }
                    break;
                    
                case DW_AT_specification:
                    if (attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                        specification_die_offset = form_value.Reference(this);
                    break;
                }
            }
            
            if (tag == DW_TAG_subprogram)
            {
                if (lo_pc != DW_INVALID_ADDRESS && hi_pc != DW_INVALID_ADDRESS)
                {
                    aranges->AppendRange (m_offset, lo_pc, hi_pc);
                }
                else
                {
                    for (size_t i=0, num_ranges = ranges.Size(); i<num_ranges; ++i)
                    {
                        const DWARFDebugRanges::Range *range = ranges.RangeAtIndex (i);
                        aranges->AppendRange (m_offset, range->begin_offset, range->end_offset);
                    }
                }
            }
        }

        die_info.die_idx = std::distance (begin, pos);

        switch (tag)
        {
        case DW_TAG_subprogram:
            if (has_address)
            {
                if (name)
                {
                    if ((name[0] == '-' || name[0] == '+') && name[1] == '[')
                    {
                        int name_len = strlen (name);
                        // Objective C methods must have at least:
                        //      "-[" or "+[" prefix
                        //      One character for a class name
                        //      One character for the space between the class name
                        //      One character for the method name
                        //      "]" suffix
                        if (name_len >= 6 && name[name_len - 1] == ']')
                        {
                            const char *method_name = strchr (name, ' ');
                            if (method_name)
                            {
                                ConstString class_name (name + 2, method_name - name - 2);
                                
                                // Keep a map of the objective C class name to all selector
                                // DIEs
                                objc_class_selectors.Insert(class_name, die_info);
                                
                                // Skip the space
                                ++method_name;
                                // Extract the objective C basename and add it to the
                                // accelerator tables
                                size_t method_name_len = name_len - (method_name - name) - 1;                                
                                func_selectors.Insert (ConstString (method_name, method_name_len), die_info);
                            }
                        }
                    }
                    // If we have a mangled name, then the DW_AT_name attribute
                    // is usually the method name without the class or any parameters
                    const DWARFDebugInfoEntry *parent = die.GetParent();
                    bool is_method = false;
                    if (parent)
                    {
                        dw_tag_t parent_tag = parent->Tag();
                        if (parent_tag == DW_TAG_class_type || parent_tag == DW_TAG_structure_type)
                        {
                            is_method = true;
                        }
                        else
                        {
                            if (mangled && specification_die_offset != DW_INVALID_OFFSET)
                            {
                                const DWARFDebugInfoEntry *specification_die = m_dwarf2Data->DebugInfo()->GetDIEPtr (specification_die_offset, NULL);
                                if (specification_die)
                                {
                                    parent = specification_die->GetParent();
                                    if (parent)
                                    {
                                        parent_tag = parent->Tag();
                                    
                                        if (parent_tag == DW_TAG_class_type || parent_tag == DW_TAG_structure_type)
                                            is_method = true;
                                    }
                                }
                            }
                        }
                    }


                    if (is_method)
                        func_methods.Insert (ConstString(name), die_info);
                    else
                        func_basenames.Insert (ConstString(name), die_info);
                }
                if (mangled.GetMangledName())
                    func_fullnames.Insert (mangled.GetMangledName(), die_info);
                if (mangled.GetDemangledName())
                    func_fullnames.Insert (mangled.GetDemangledName(), die_info);
            }
            break;

        case DW_TAG_inlined_subroutine:
            if (has_address)
            {
                if (name)
                    func_basenames.Insert (ConstString(name), die_info);
                if (mangled.GetMangledName())
                    func_fullnames.Insert (mangled.GetMangledName(), die_info);
                if (mangled.GetDemangledName())
                    func_fullnames.Insert (mangled.GetDemangledName(), die_info);
            }
            break;
        
        case DW_TAG_base_type:
        case DW_TAG_class_type:
        case DW_TAG_constant:
        case DW_TAG_enumeration_type:
        case DW_TAG_string_type:
        case DW_TAG_subroutine_type:
        case DW_TAG_structure_type:
        case DW_TAG_union_type:
        case DW_TAG_typedef:
            if (name && is_declaration == false)
            {
                types.Insert (ConstString(name), die_info);
            }
            break;

        case DW_TAG_namespace:
            if (name)
                namespaces.Insert (ConstString(name), die_info);
            break;

        case DW_TAG_variable:
            if (name && has_location && is_global_or_static_variable)
            {
                globals.Insert (ConstString(name), die_info);
                // Be sure to include variables by their mangled and demangled
                // names if they have any since a variable can have a basename
                // "i", a mangled named "_ZN12_GLOBAL__N_11iE" and a demangled 
                // mangled name "(anonymous namespace)::i"...
                if (mangled.GetMangledName())
                    globals.Insert (mangled.GetMangledName(), die_info);
                if (mangled.GetDemangledName())
                    globals.Insert (mangled.GetDemangledName(), die_info);
            }
            break;
            
        default:
            continue;
        }
    }
}


