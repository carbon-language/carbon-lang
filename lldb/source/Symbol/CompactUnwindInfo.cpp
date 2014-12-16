//===-- CompactUnwindInfo.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// C Includes
// C++ Includes
#include <algorithm>

#include "lldb/Core/Log.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/CompactUnwindInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/UnwindPlan.h"

#include "llvm/Support/MathExtras.h"

using namespace lldb;
using namespace lldb_private;


namespace lldb_private {

    // Constants from <mach-o/compact_unwind_encoding.h>

    enum {
        UNWIND_IS_NOT_FUNCTION_START           = 0x80000000,
        UNWIND_HAS_LSDA                        = 0x40000000,
        UNWIND_PERSONALITY_MASK                = 0x30000000,
    };

    enum {
        UNWIND_X86_MODE_MASK                         = 0x0F000000,
        UNWIND_X86_MODE_EBP_FRAME                    = 0x01000000,
        UNWIND_X86_MODE_STACK_IMMD                   = 0x02000000,
        UNWIND_X86_MODE_STACK_IND                    = 0x03000000,
        UNWIND_X86_MODE_DWARF                        = 0x04000000,

        UNWIND_X86_EBP_FRAME_REGISTERS               = 0x00007FFF,
        UNWIND_X86_EBP_FRAME_OFFSET                  = 0x00FF0000,

        UNWIND_X86_FRAMELESS_STACK_SIZE              = 0x00FF0000,
        UNWIND_X86_FRAMELESS_STACK_ADJUST            = 0x0000E000,
        UNWIND_X86_FRAMELESS_STACK_REG_COUNT         = 0x00001C00,
        UNWIND_X86_FRAMELESS_STACK_REG_PERMUTATION   = 0x000003FF,

        UNWIND_X86_DWARF_SECTION_OFFSET              = 0x00FFFFFF,
    };

    enum {
        UNWIND_X86_REG_NONE     = 0,
        UNWIND_X86_REG_EBX      = 1,
        UNWIND_X86_REG_ECX      = 2,
        UNWIND_X86_REG_EDX      = 3,
        UNWIND_X86_REG_EDI      = 4,
        UNWIND_X86_REG_ESI      = 5,
        UNWIND_X86_REG_EBP      = 6,
    };
    enum {
        UNWIND_X86_64_MODE_MASK                         = 0x0F000000,
        UNWIND_X86_64_MODE_RBP_FRAME                    = 0x01000000,
        UNWIND_X86_64_MODE_STACK_IMMD                   = 0x02000000,
        UNWIND_X86_64_MODE_STACK_IND                    = 0x03000000,
        UNWIND_X86_64_MODE_DWARF                        = 0x04000000,

        UNWIND_X86_64_RBP_FRAME_REGISTERS               = 0x00007FFF,
        UNWIND_X86_64_RBP_FRAME_OFFSET                  = 0x00FF0000,

        UNWIND_X86_64_FRAMELESS_STACK_SIZE              = 0x00FF0000,
        UNWIND_X86_64_FRAMELESS_STACK_ADJUST            = 0x0000E000,
        UNWIND_X86_64_FRAMELESS_STACK_REG_COUNT         = 0x00001C00,
        UNWIND_X86_64_FRAMELESS_STACK_REG_PERMUTATION   = 0x000003FF,

        UNWIND_X86_64_DWARF_SECTION_OFFSET              = 0x00FFFFFF,
    };

    enum {
        UNWIND_X86_64_REG_NONE       = 0,
        UNWIND_X86_64_REG_RBX        = 1,
        UNWIND_X86_64_REG_R12        = 2,
        UNWIND_X86_64_REG_R13        = 3,
        UNWIND_X86_64_REG_R14        = 4,
        UNWIND_X86_64_REG_R15        = 5,
        UNWIND_X86_64_REG_RBP        = 6,
    };
};


#ifndef UNWIND_SECOND_LEVEL_REGULAR
#define UNWIND_SECOND_LEVEL_REGULAR 2
#endif

#ifndef UNWIND_SECOND_LEVEL_COMPRESSED
#define UNWIND_SECOND_LEVEL_COMPRESSED 3
#endif

#ifndef UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET
#define UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET(entry)            (entry & 0x00FFFFFF)
#endif

#ifndef UNWIND_INFO_COMPRESSED_ENTRY_ENCODING_INDEX
#define UNWIND_INFO_COMPRESSED_ENTRY_ENCODING_INDEX(entry)        ((entry >> 24) & 0xFF)
#endif

#define EXTRACT_BITS(value, mask) \
        ( (value >> llvm::countTrailingZeros(static_cast<uint32_t>(mask), llvm::ZB_Width)) & \
          (((1 << llvm::CountPopulation_32(static_cast<uint32_t>(mask))))-1) )



//----------------------
// constructor 
//----------------------


CompactUnwindInfo::CompactUnwindInfo(ObjectFile& objfile, SectionSP& section_sp) :
    m_objfile (objfile),
    m_section_sp (section_sp),
    m_mutex (),
    m_indexes (),
    m_indexes_computed (eLazyBoolCalculate),
    m_unwindinfo_data (),
    m_unwindinfo_data_computed (false),
    m_unwind_header ()
{

}

//----------------------
// destructor
//----------------------

CompactUnwindInfo::~CompactUnwindInfo()
{
}

bool
CompactUnwindInfo::GetUnwindPlan (Target &target, Address addr, UnwindPlan& unwind_plan)
{
    if (!IsValid ())
    {
        return false;
    }
    FunctionInfo function_info;
    if (GetCompactUnwindInfoForFunction (target, addr, function_info))
    {
        // shortcut return for functions that have no compact unwind
        if (function_info.encoding == 0)
            return false;

        ArchSpec arch;
        if (m_objfile.GetArchitecture (arch))
        {
            if (arch.GetTriple().getArch() == llvm::Triple::x86_64)
            {
                return CreateUnwindPlan_x86_64 (target, function_info, unwind_plan, addr);
            }
            if (arch.GetTriple().getArch() == llvm::Triple::x86)
            {
                return CreateUnwindPlan_i386 (target, function_info, unwind_plan, addr);
            }
        }
    }
    return false;
}

bool
CompactUnwindInfo::IsValid ()
{
    if (m_section_sp.get() == nullptr || m_section_sp->IsEncrypted())
        return false;

    if (m_indexes_computed == eLazyBoolYes && m_unwindinfo_data_computed)
        return true;

    ScanIndex ();

    return m_indexes_computed == eLazyBoolYes && m_unwindinfo_data_computed;
}

void
CompactUnwindInfo::ScanIndex ()
{
    Mutex::Locker locker(m_mutex);
    if (m_indexes_computed == eLazyBoolYes && m_unwindinfo_data_computed)
        return;

    // We can't read the index for some reason.
    if (m_indexes_computed == eLazyBoolNo)
    {
        return;
    }

    if (m_unwindinfo_data_computed == false)
    {
        m_objfile.ReadSectionData (m_section_sp.get(), m_unwindinfo_data);
        m_unwindinfo_data_computed = true;
    }

    if (m_unwindinfo_data.GetByteSize() > 0)
    {
        offset_t offset = 0;

                // struct unwind_info_section_header
                // {
                // uint32_t    version;            // UNWIND_SECTION_VERSION
                // uint32_t    commonEncodingsArraySectionOffset;
                // uint32_t    commonEncodingsArrayCount;
                // uint32_t    personalityArraySectionOffset;
                // uint32_t    personalityArrayCount;
                // uint32_t    indexSectionOffset;
                // uint32_t    indexCount;
        
        m_unwind_header.version = m_unwindinfo_data.GetU32(&offset);
        m_unwind_header.common_encodings_array_offset = m_unwindinfo_data.GetU32(&offset);
        m_unwind_header.common_encodings_array_count = m_unwindinfo_data.GetU32(&offset);
        m_unwind_header.personality_array_offset = m_unwindinfo_data.GetU32(&offset);
        m_unwind_header.personality_array_count = m_unwindinfo_data.GetU32(&offset);
        uint32_t indexSectionOffset = m_unwindinfo_data.GetU32(&offset);

        uint32_t indexCount = m_unwindinfo_data.GetU32(&offset);

        if (m_unwind_header.version != 1)
        {
            m_indexes_computed = eLazyBoolNo;
        }

        // Parse the basic information from the indexes
        // We wait to scan the second level page info until it's needed

            // struct unwind_info_section_header_index_entry 
            // {
            //     uint32_t        functionOffset;
            //     uint32_t        secondLevelPagesSectionOffset;
            //     uint32_t        lsdaIndexArraySectionOffset;
            // };

        offset = indexSectionOffset;
        for (int idx = 0; idx < indexCount; idx++)
        {
            uint32_t function_offset = m_unwindinfo_data.GetU32(&offset);      // functionOffset
            uint32_t second_level_offset = m_unwindinfo_data.GetU32(&offset);  // secondLevelPagesSectionOffset
            uint32_t lsda_offset = m_unwindinfo_data.GetU32(&offset);          // lsdaIndexArraySectionOffset

            if (second_level_offset > m_section_sp->GetByteSize() || lsda_offset > m_section_sp->GetByteSize())
            {
                m_indexes_computed = eLazyBoolNo;
            }

            UnwindIndex this_index;
            this_index.function_offset = function_offset;     // 
            this_index.second_level = second_level_offset;
            this_index.lsda_array_start = lsda_offset;

            if (m_indexes.size() > 0)
            {
                m_indexes[m_indexes.size() - 1].lsda_array_end = lsda_offset;
            }

            if (second_level_offset == 0)
            {
                this_index.sentinal_entry = true;
            }

            m_indexes.push_back (this_index);
        }
        m_indexes_computed = eLazyBoolYes;
    }
    else
    {
        m_indexes_computed = eLazyBoolNo;
    }
}

uint32_t
CompactUnwindInfo::GetLSDAForFunctionOffset (uint32_t lsda_offset, uint32_t lsda_count, uint32_t function_offset)
{
        // struct unwind_info_section_header_lsda_index_entry 
        // {
        //         uint32_t        functionOffset;
        //         uint32_t        lsdaOffset;
        // };

    offset_t first_entry = lsda_offset;
    uint32_t low = 0;
    uint32_t high = lsda_count;
    while (low < high)
    {
        uint32_t mid = (low + high) / 2;
        offset_t offset = first_entry + (mid * 8);
        uint32_t mid_func_offset = m_unwindinfo_data.GetU32(&offset);  // functionOffset
        uint32_t mid_lsda_offset = m_unwindinfo_data.GetU32(&offset);  // lsdaOffset
        if (mid_func_offset == function_offset)
        {
            return mid_lsda_offset;
        }
        if (mid_func_offset < function_offset)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return 0;
}

lldb::offset_t
CompactUnwindInfo::BinarySearchRegularSecondPage (uint32_t entry_page_offset, uint32_t entry_count, uint32_t function_offset)
{
    // typedef uint32_t compact_unwind_encoding_t;
    // struct unwind_info_regular_second_level_entry 
    // {
    //     uint32_t                    functionOffset;
    //     compact_unwind_encoding_t    encoding;

    offset_t first_entry = entry_page_offset;

    uint32_t low = 0;
    uint32_t high = entry_count;
    uint32_t last = high - 1;
    while (low < high)
    {
        uint32_t mid = (low + high) / 2;
        offset_t offset = first_entry + (mid * 8);
        uint32_t mid_func_offset = m_unwindinfo_data.GetU32(&offset);   // functionOffset
        uint32_t next_func_offset = 0;
        if (mid < last)
        {
            offset = first_entry + ((mid + 1) * 8);
            next_func_offset = m_unwindinfo_data.GetU32(&offset);       // functionOffset
        }
        if (mid_func_offset <= function_offset)
        {
            if (mid == last || (next_func_offset > function_offset))
            {
                return first_entry + (mid * 8);
            }
            else
            {
                low = mid + 1;
            }
        }
        else
        {
            high = mid;
        }
    }
    return LLDB_INVALID_OFFSET;
}

uint32_t
CompactUnwindInfo::BinarySearchCompressedSecondPage (uint32_t entry_page_offset, uint32_t entry_count, uint32_t function_offset_to_find, uint32_t function_offset_base)
{
    offset_t first_entry = entry_page_offset;

    uint32_t low = 0;
    uint32_t high = entry_count;
    uint32_t last = high - 1;
    while (low < high)
    {
        uint32_t mid = (low + high) / 2;
        offset_t offset = first_entry + (mid * 4);
        uint32_t entry = m_unwindinfo_data.GetU32(&offset);   // entry
        uint32_t mid_func_offset = UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET (entry);
        mid_func_offset += function_offset_base;
        uint32_t next_func_offset = 0;
        if (mid < last)
        {
            offset = first_entry + ((mid + 1) * 4);
            uint32_t next_entry = m_unwindinfo_data.GetU32(&offset);       // entry
            next_func_offset = UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET (next_entry);
            next_func_offset += function_offset_base;
        }
        if (mid_func_offset <= function_offset_to_find)
        {
            if (mid == last || (next_func_offset > function_offset_to_find))
            {
                return UNWIND_INFO_COMPRESSED_ENTRY_ENCODING_INDEX (entry);
            }
            else
            {
                low = mid + 1;
            }
        }
        else
        {
            high = mid;
        }
    }

    return UINT32_MAX;
}


bool
CompactUnwindInfo::GetCompactUnwindInfoForFunction (Target &target, Address address, FunctionInfo &unwind_info)
{
    unwind_info.encoding = 0;
    unwind_info.lsda_address.Clear();
    unwind_info.personality_ptr_address.Clear();

    if (!IsValid ())
        return false;

    // FIXME looking into a problem with getting the wrong compact unwind entry for
    // _CFRunLoopRun from CoreFoundation in a live process; disabling the Compact 
    // Unwind plans until I get to the bottom of what's going on there.
    return false;

    addr_t text_section_file_address = LLDB_INVALID_ADDRESS;
    SectionList *sl = m_objfile.GetSectionList ();
    if (sl)
    {
        SectionSP text_sect = sl->FindSectionByType (eSectionTypeCode, true);
        if (text_sect.get())
        {
           text_section_file_address = text_sect->GetFileAddress();
        }
    }
    if (text_section_file_address == LLDB_INVALID_ADDRESS)
        return false;

    addr_t function_offset = address.GetFileAddress() - m_objfile.GetHeaderAddress().GetFileAddress();
    
    UnwindIndex key;
    key.function_offset = function_offset;
    
    std::vector<UnwindIndex>::const_iterator it;
    it = std::lower_bound (m_indexes.begin(), m_indexes.end(), key);
    if (it == m_indexes.end())
    {
        return false;
    }

    if (it->function_offset != key.function_offset)
    {
        if (it != m_indexes.begin())
            --it;
    }

    if (it->sentinal_entry == true)
    {
        return false;
    }

    offset_t second_page_offset = it->second_level;
    offset_t lsda_array_start = it->lsda_array_start;
    offset_t lsda_array_count = (it->lsda_array_end - it->lsda_array_start) / 8;

    offset_t offset = second_page_offset;
    uint32_t kind = m_unwindinfo_data.GetU32(&offset);  // UNWIND_SECOND_LEVEL_REGULAR or UNWIND_SECOND_LEVEL_COMPRESSED

    if (kind == UNWIND_SECOND_LEVEL_REGULAR)
    {
            // struct unwind_info_regular_second_level_page_header
            // {
            //     uint32_t    kind;    // UNWIND_SECOND_LEVEL_REGULAR
            //     uint16_t    entryPageOffset;
            //     uint16_t    entryCount;

            // typedef uint32_t compact_unwind_encoding_t;
            // struct unwind_info_regular_second_level_entry 
            // {
            //     uint32_t                    functionOffset;
            //     compact_unwind_encoding_t    encoding;

        uint16_t entry_page_offset = m_unwindinfo_data.GetU16(&offset); // entryPageOffset
        uint16_t entry_count = m_unwindinfo_data.GetU16(&offset);       // entryCount

        offset_t entry_offset = BinarySearchRegularSecondPage (second_page_offset + entry_page_offset, entry_count, function_offset);
        if (entry_offset == LLDB_INVALID_OFFSET)
        {
            return false;
        }
        entry_offset += 4;                                              // skip over functionOffset
        unwind_info.encoding = m_unwindinfo_data.GetU32(&entry_offset); // encoding
        if (unwind_info.encoding & UNWIND_HAS_LSDA)
        {
            SectionList *sl = m_objfile.GetSectionList ();
            if (sl)
            {
                uint32_t lsda_offset = GetLSDAForFunctionOffset (lsda_array_start, lsda_array_count, function_offset);
                addr_t objfile_header_file_address = m_objfile.GetHeaderAddress().GetFileAddress();
                unwind_info.lsda_address.ResolveAddressUsingFileSections (objfile_header_file_address + lsda_offset, sl);
            }
        }
        if (unwind_info.encoding & UNWIND_PERSONALITY_MASK)
        {
            uint32_t personality_index = EXTRACT_BITS (unwind_info.encoding, UNWIND_PERSONALITY_MASK);

            if (personality_index > 0)
            {
                personality_index--;
                if (personality_index < m_unwind_header.personality_array_count)
                {
                    offset_t offset = m_unwind_header.personality_array_offset;
                    offset += 4 * personality_index;
                    SectionList *sl = m_objfile.GetSectionList ();
                    if (sl)
                    {
                        uint32_t personality_offset = m_unwindinfo_data.GetU32(&offset);
                        addr_t objfile_header_file_address = m_objfile.GetHeaderAddress().GetFileAddress();
                        unwind_info.personality_ptr_address.ResolveAddressUsingFileSections (objfile_header_file_address + personality_offset, sl);
                    }
                }
            }
        }
        return true;
    }
    else if (kind == UNWIND_SECOND_LEVEL_COMPRESSED)
    {
            // struct unwind_info_compressed_second_level_page_header
            // {
            //     uint32_t    kind;    // UNWIND_SECOND_LEVEL_COMPRESSED
            //     uint16_t    entryPageOffset;         // offset from this 2nd lvl page idx to array of entries
            //                                          // (an entry has a function offset and index into the encodings)
            //                                          // NB function offset from the entry in the compressed page 
            //                                          // must be added to the index's functionOffset value.
            //     uint16_t    entryCount;             
            //     uint16_t    encodingsPageOffset;     // offset from this 2nd lvl page idx to array of encodings
            //     uint16_t    encodingsCount;

        uint16_t entry_page_offset = m_unwindinfo_data.GetU16(&offset);     // entryPageOffset
        uint16_t entry_count = m_unwindinfo_data.GetU16(&offset);           // entryCount
        uint16_t encodings_page_offset = m_unwindinfo_data.GetU16(&offset); // encodingsPageOffset
        uint16_t encodings_count = m_unwindinfo_data.GetU16(&offset);       // encodingsCount

        uint32_t encoding_index = BinarySearchCompressedSecondPage (second_page_offset + entry_page_offset, entry_count, function_offset, it->function_offset);
        if (encoding_index == UINT32_MAX || encoding_index >= encodings_count + m_unwind_header.common_encodings_array_count)
        {
            return false;
        }
        uint32_t encoding = 0;
        if (encoding_index < m_unwind_header.common_encodings_array_count)
        {
            offset = m_unwind_header.common_encodings_array_offset + (encoding_index * sizeof (uint32_t));
            encoding = m_unwindinfo_data.GetU32(&offset);   // encoding entry from the commonEncodingsArray
        }
        else 
        {
            uint32_t page_specific_entry_index = encoding_index - m_unwind_header.common_encodings_array_count;
            offset = second_page_offset + encodings_page_offset + (page_specific_entry_index * sizeof (uint32_t));
            encoding = m_unwindinfo_data.GetU32(&offset);   // encoding entry from the page-specific encoding array
        }
        if (encoding == 0)
            return false;
        unwind_info.encoding = encoding;

        unwind_info.encoding = encoding;
        if (unwind_info.encoding & UNWIND_HAS_LSDA)
        {
            SectionList *sl = m_objfile.GetSectionList ();
            if (sl)
            {
                uint32_t lsda_offset = GetLSDAForFunctionOffset (lsda_array_start, lsda_array_count, function_offset);
                addr_t objfile_header_file_address = m_objfile.GetHeaderAddress().GetFileAddress();
                unwind_info.lsda_address.ResolveAddressUsingFileSections (objfile_header_file_address + lsda_offset, sl);
            }
        }
        if (unwind_info.encoding & UNWIND_PERSONALITY_MASK)
        {
            uint32_t personality_index = EXTRACT_BITS (unwind_info.encoding, UNWIND_PERSONALITY_MASK);

            if (personality_index > 0)
            {
                personality_index--;
                if (personality_index < m_unwind_header.personality_array_count)
                {
                    offset_t offset = m_unwind_header.personality_array_offset;
                    offset += 4 * personality_index;
                    SectionList *sl = m_objfile.GetSectionList ();
                    if (sl)
                    {
                        uint32_t personality_offset = m_unwindinfo_data.GetU32(&offset);
                        addr_t objfile_header_file_address = m_objfile.GetHeaderAddress().GetFileAddress();
                        unwind_info.personality_ptr_address.ResolveAddressUsingFileSections (objfile_header_file_address + personality_offset, sl);
                    }
                }
            }
        }
        return true;
    }
    return false;
}

enum x86_64_eh_regnum {
    rax = 0,
    rdx = 1,
    rcx = 2,
    rbx = 3,
    rsi = 4,
    rdi = 5,
    rbp = 6,
    rsp = 7,
    r8 = 8,
    r9 = 9,
    r10 = 10,
    r11 = 11,
    r12 = 12,
    r13 = 13,
    r14 = 14,
    r15 = 15,
    rip = 16   // this is officially the Return Address register number, but close enough
};

// Convert the compact_unwind_info.h register numbering scheme
// to eRegisterKindGCC (eh_frame) register numbering scheme.
uint32_t
translate_to_eh_frame_regnum_x86_64 (uint32_t unwind_regno)
{
    switch (unwind_regno)
    {
        case UNWIND_X86_64_REG_RBX:
            return x86_64_eh_regnum::rbx;
        case UNWIND_X86_64_REG_R12:
            return x86_64_eh_regnum::r12;
        case UNWIND_X86_64_REG_R13:
            return x86_64_eh_regnum::r13;
        case UNWIND_X86_64_REG_R14:
            return x86_64_eh_regnum::r14;
        case UNWIND_X86_64_REG_R15:
            return x86_64_eh_regnum::r15;
        case UNWIND_X86_64_REG_RBP:
            return x86_64_eh_regnum::rbp;
        default:
            return LLDB_INVALID_REGNUM;
    }
}

bool
CompactUnwindInfo::CreateUnwindPlan_x86_64 (Target &target, FunctionInfo &function_info, UnwindPlan &unwind_plan, Address pc_or_function_start)
{
    unwind_plan.SetSourceName ("compact unwind info");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolYes);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolNo);
    unwind_plan.SetRegisterKind (eRegisterKindGCC);

    unwind_plan.SetLSDAAddress (function_info.lsda_address);
    unwind_plan.SetPersonalityFunctionPtr (function_info.personality_ptr_address);

    UnwindPlan::RowSP row (new UnwindPlan::Row);

    const int wordsize = 8;
    int mode = function_info.encoding & UNWIND_X86_64_MODE_MASK;
    switch (mode)
    {
        case UNWIND_X86_64_MODE_RBP_FRAME:
        {
            row->SetCFARegister (translate_to_eh_frame_regnum_x86_64 (UNWIND_X86_64_REG_RBP));
            row->SetCFAOffset (2 * wordsize);
            row->SetOffset (0);
            row->SetRegisterLocationToAtCFAPlusOffset (x86_64_eh_regnum::rbp, wordsize * -2, true);
            row->SetRegisterLocationToAtCFAPlusOffset (x86_64_eh_regnum::rip, wordsize * -1, true);
            row->SetRegisterLocationToIsCFAPlusOffset (x86_64_eh_regnum::rsp, 0, true);
            
            uint32_t saved_registers_offset = EXTRACT_BITS (function_info.encoding, UNWIND_X86_64_RBP_FRAME_OFFSET);

            uint32_t saved_registers_locations = EXTRACT_BITS (function_info.encoding, UNWIND_X86_64_RBP_FRAME_REGISTERS);

            saved_registers_offset += 2;

            for (int i = 0; i < 5; i++)
            {
                uint32_t regnum = saved_registers_locations & 0x7;
                switch (regnum)
                {
                    case UNWIND_X86_64_REG_NONE:
                        break;
                    case UNWIND_X86_64_REG_RBX:
                    case UNWIND_X86_64_REG_R12:
                    case UNWIND_X86_64_REG_R13:
                    case UNWIND_X86_64_REG_R14:
                    case UNWIND_X86_64_REG_R15:
                        row->SetRegisterLocationToAtCFAPlusOffset (translate_to_eh_frame_regnum_x86_64 (regnum), wordsize * -saved_registers_offset, true);
                        break;
                }
                saved_registers_offset--;
                saved_registers_locations >>= 3;
            }
            unwind_plan.AppendRow (row);
            return true;
        }
        break;

        case UNWIND_X86_64_MODE_STACK_IND:
        {
            // The clang in Xcode 6 is emitting incorrect compact unwind encodings for this
            // style of unwind.  It was fixed in llvm r217020 although the algorith being
            // used to compute this style of unwind in generateCompactUnwindEncodingImpl()
            // isn't as foolproof as I'm comfortable with -- if any instructions other than
            // a push are scheduled before the subq, it will give bogus encoding results.

            // The target and pc_or_function_start arguments will be needed to handle this
            // encoding style correctly -- to find the start address of the function and 
            // read memory offset from there.
            return false;
        }
        break;

#if 0
        case UNWIND_X86_64_MODE_STACK_IMMD:
        {
            uint32_t stack_size = EXTRACT_BITS (encoding, UNWIND_X86_64_FRAMELESS_STACK_SIZE);
            uint32_t register_count = EXTRACT_BITS (encoding, UNWIND_X86_64_FRAMELESS_STACK_REG_COUNT);
            uint32_t permutation = EXTRACT_BITS (encoding, UNWIND_X86_64_FRAMELESS_STACK_REG_PERMUTATION);

            if (mode == UNWIND_X86_64_MODE_STACK_IND && function_start)
            {
                uint32_t stack_adjust = EXTRACT_BITS (encoding, UNWIND_X86_64_FRAMELESS_STACK_ADJUST);

                // offset into the function instructions; 0 == beginning of first instruction
                uint32_t offset_to_subl_insn = EXTRACT_BITS (encoding, UNWIND_X86_64_FRAMELESS_STACK_SIZE);

                stack_size = *((uint32_t*) (function_start + offset_to_subl_insn));

                stack_size += stack_adjust * 8;

                printf ("large stack ");
            }
            
            printf ("frameless function: stack size %d, register count %d ", stack_size * 8, register_count);

            if (register_count == 0)
            {
                printf (" no registers saved");
            }
            else
            {

                // We need to include (up to) 6 registers in 10 bits.
                // That would be 18 bits if we just used 3 bits per reg to indicate
                // the order they're saved on the stack. 
                //
                // This is done with Lehmer code permutation, e.g. see
                // http://stackoverflow.com/questions/1506078/fast-permutation-number-permutation-mapping-algorithms
                int permunreg[6];

                // This decodes the variable-base number in the 10 bits
                // and gives us the Lehmer code sequence which can then
                // be decoded.

                switch (register_count) 
                {
                    case 6:
                        permunreg[0] = permutation/120;    // 120 == 5!
                        permutation -= (permunreg[0]*120);
                        permunreg[1] = permutation/24;     // 24 == 4!
                        permutation -= (permunreg[1]*24);
                        permunreg[2] = permutation/6;      // 6 == 3!
                        permutation -= (permunreg[2]*6);
                        permunreg[3] = permutation/2;      // 2 == 2!
                        permutation -= (permunreg[3]*2);
                        permunreg[4] = permutation;        // 1 == 1!
                        permunreg[5] = 0;
                        break;
                    case 5:
                        permunreg[0] = permutation/120;
                        permutation -= (permunreg[0]*120);
                        permunreg[1] = permutation/24;
                        permutation -= (permunreg[1]*24);
                        permunreg[2] = permutation/6;
                        permutation -= (permunreg[2]*6);
                        permunreg[3] = permutation/2;
                        permutation -= (permunreg[3]*2);
                        permunreg[4] = permutation;
                        break;
                    case 4:
                        permunreg[0] = permutation/60;
                        permutation -= (permunreg[0]*60);
                        permunreg[1] = permutation/12;
                        permutation -= (permunreg[1]*12);
                        permunreg[2] = permutation/3;
                        permutation -= (permunreg[2]*3);
                        permunreg[3] = permutation;
                        break;
                    case 3:
                        permunreg[0] = permutation/20;
                        permutation -= (permunreg[0]*20);
                        permunreg[1] = permutation/4;
                        permutation -= (permunreg[1]*4);
                        permunreg[2] = permutation;
                        break;
                    case 2:
                        permunreg[0] = permutation/5;
                        permutation -= (permunreg[0]*5);
                        permunreg[1] = permutation;
                        break;
                    case 1:
                        permunreg[0] = permutation;
                        break;
                }
                
                // Decode the Lehmer code for this permutation of
                // the registers v. http://en.wikipedia.org/wiki/Lehmer_code

                int registers[6];
                bool used[7] = { false, false, false, false, false, false, false };
                for (int i = 0; i < register_count; i++)
                {
                    int renum = 0;
                    for (int j = 1; j < 7; j++)
                    {
                        if (used[j] == false)
                        {
                            if (renum == permunreg[i])
                            {
                                registers[i] = j;
                                used[j] = true;
                                break;
                            }
                            renum++;
                        }
                    }
                }


                printf (" CFA is rsp+%d ", stack_size * 8);

                uint32_t saved_registers_offset = 1;
                printf (" rip=[CFA-%d]", saved_registers_offset * 8);
                saved_registers_offset++;

                for (int i = (sizeof (registers) / sizeof (int)) - 1; i >= 0; i--)
                {
                    switch (registers[i])
                    {
                        case UNWIND_X86_64_REG_NONE:
                            break;
                        case UNWIND_X86_64_REG_RBX:
                            printf (" rbx=[CFA-%d]", saved_registers_offset * 8);
                            break;
                        case UNWIND_X86_64_REG_R12:
                            printf (" r12=[CFA-%d]", saved_registers_offset * 8);
                            break;
                        case UNWIND_X86_64_REG_R13:
                            printf (" r13=[CFA-%d]", saved_registers_offset * 8);
                            break;
                        case UNWIND_X86_64_REG_R14:
                            printf (" r14=[CFA-%d]", saved_registers_offset * 8);
                            break;
                        case UNWIND_X86_64_REG_R15:
                            printf (" r15=[CFA-%d]", saved_registers_offset * 8);
                            break;
                        case UNWIND_X86_64_REG_RBP:
                            printf (" rbp=[CFA-%d]", saved_registers_offset * 8);
                            break;
                    }
                    saved_registers_offset++;
                }

            }

        }
        break;
#endif

        case UNWIND_X86_64_MODE_DWARF:
        {
            return false;
        }
        break;

        case 0:
        {
            return false;
        }
        break;
    }
    return false;
}

enum i386_eh_regnum {
    eax = 0,
    ecx = 1,
    edx = 2,
    ebx = 3,
    ebp = 4,
    esp = 5,
    esi = 6,
    edi = 7,
    eip = 8    // this is officially the Return Address register number, but close enough
};

// Convert the compact_unwind_info.h register numbering scheme
// to eRegisterKindGCC (eh_frame) register numbering scheme.
uint32_t
translate_to_eh_frame_regnum_i386 (uint32_t unwind_regno)
{
    switch (unwind_regno)
    {
        case UNWIND_X86_REG_EBX:
            return i386_eh_regnum::ebx;
        case UNWIND_X86_REG_ECX:
            return i386_eh_regnum::ecx;
        case UNWIND_X86_REG_EDX:
            return i386_eh_regnum::edx;
        case UNWIND_X86_REG_EDI:
            return i386_eh_regnum::edi;
        case UNWIND_X86_REG_ESI:
            return i386_eh_regnum::esi;
        case UNWIND_X86_REG_EBP:
            return i386_eh_regnum::ebp;
        default:
            return LLDB_INVALID_REGNUM;
    }
}


bool
CompactUnwindInfo::CreateUnwindPlan_i386 (Target &target, FunctionInfo &function_info, UnwindPlan &unwind_plan, Address pc_or_function_start)
{
    unwind_plan.SetSourceName ("compact unwind info");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolYes);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolNo);
    unwind_plan.SetRegisterKind (eRegisterKindGCC);

    unwind_plan.SetLSDAAddress (function_info.lsda_address);
    unwind_plan.SetPersonalityFunctionPtr (function_info.personality_ptr_address);

    UnwindPlan::RowSP row (new UnwindPlan::Row);

    const int wordsize = 4;
    int mode = function_info.encoding & UNWIND_X86_MODE_MASK;
    switch (mode)
    {
        case UNWIND_X86_MODE_EBP_FRAME:
        {
            row->SetCFARegister (translate_to_eh_frame_regnum_i386 (UNWIND_X86_REG_EBP));
            row->SetCFAOffset (2 * wordsize);
            row->SetOffset (0);
            row->SetRegisterLocationToAtCFAPlusOffset (i386_eh_regnum::ebp, wordsize * -2, true);
            row->SetRegisterLocationToAtCFAPlusOffset (i386_eh_regnum::eip, wordsize * -1, true);
            row->SetRegisterLocationToIsCFAPlusOffset (i386_eh_regnum::esp, 0, true);
            
            uint32_t saved_registers_offset = EXTRACT_BITS (function_info.encoding, UNWIND_X86_EBP_FRAME_OFFSET);

            uint32_t saved_registers_locations = EXTRACT_BITS (function_info.encoding, UNWIND_X86_EBP_FRAME_REGISTERS);

            saved_registers_offset += 2;

            for (int i = 0; i < 5; i++)
            {
                uint32_t regnum = saved_registers_locations & 0x7;
                switch (regnum)
                {
                    case UNWIND_X86_REG_NONE:
                        break;
                    case UNWIND_X86_REG_EBX:
                    case UNWIND_X86_REG_ECX:
                    case UNWIND_X86_REG_EDX:
                    case UNWIND_X86_REG_EDI:
                    case UNWIND_X86_REG_ESI:
                        row->SetRegisterLocationToAtCFAPlusOffset (translate_to_eh_frame_regnum_i386 (regnum), wordsize * -saved_registers_offset, true);
                        break;
                }
                saved_registers_offset--;
                saved_registers_locations >>= 3;
            }
            unwind_plan.AppendRow (row);
            return true;
        }
        break;

        case UNWIND_X86_MODE_STACK_IND:
        case UNWIND_X86_MODE_STACK_IMMD:
        case UNWIND_X86_MODE_DWARF:
        {
            return false;
        }
        break;
    }
    return false;
}
