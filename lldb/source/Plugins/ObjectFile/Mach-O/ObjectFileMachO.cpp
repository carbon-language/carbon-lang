//===-- ObjectFileMachO.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjectFileMachO.h"

#include <mach-o/nlist.h>
#include <mach-o/stab.h>

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/UUID.h"
#include "lldb/Symbol/ObjectFile.h"

#ifndef S_DTRACE_DOF
// section contains DTrace Object Format
#define S_DTRACE_DOF 0xf
#endif

#ifndef S_LAZY_DYLIB_SYMBOL_POINTERS
// section with only lazy symbol pointers to lazy loaded dylibs
#define S_LAZY_DYLIB_SYMBOL_POINTERS 0x10
#endif

using namespace lldb;
using namespace lldb_private;


void
ObjectFileMachO::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
ObjectFileMachO::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
ObjectFileMachO::GetPluginNameStatic()
{
    return "object-file.mach-o";
}

const char *
ObjectFileMachO::GetPluginDescriptionStatic()
{
    return "Mach-o object file reader (32 and 64 bit)";
}


ObjectFile *
ObjectFileMachO::CreateInstance (Module* module, DataBufferSP& dataSP, const FileSpec* file, addr_t offset, addr_t length)
{
    if (ObjectFileMachO::MagicBytesMatch(dataSP))
    {
        std::auto_ptr<ObjectFile> objfile_ap(new ObjectFileMachO (module, dataSP, file, offset, length));
        if (objfile_ap.get() && objfile_ap->ParseHeader())
            return objfile_ap.release();
    }
    return NULL;
}


static uint32_t
MachHeaderSizeFromMagic(uint32_t magic)
{
    switch (magic)
    {
    case MH_MAGIC:
    case MH_CIGAM:
        return sizeof(struct mach_header);

    case MH_MAGIC_64:
    case MH_CIGAM_64:
        return sizeof(struct mach_header_64);
        break;

    default:
        break;
    }
    return 0;
}


bool
ObjectFileMachO::MagicBytesMatch (DataBufferSP& dataSP)
{
    DataExtractor data(dataSP, eByteOrderHost, 4);
    uint32_t offset = 0;
    uint32_t magic = data.GetU32(&offset);
    return MachHeaderSizeFromMagic(magic) != 0;
}


ObjectFileMachO::ObjectFileMachO(Module* module, DataBufferSP& dataSP, const FileSpec* file, addr_t offset, addr_t length) :
    ObjectFile(module, file, offset, length, dataSP),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_header(),
    m_sections_ap(),
    m_symtab_ap()
{
    ::bzero (&m_header, sizeof(m_header));
    ::bzero (&m_dysymtab, sizeof(m_dysymtab));
}


ObjectFileMachO::~ObjectFileMachO()
{
}


bool
ObjectFileMachO::ParseHeader ()
{
    lldb_private::Mutex::Locker locker(m_mutex);
    bool can_parse = false;
    uint32_t offset = 0;
    m_data.SetByteOrder (eByteOrderHost);
    // Leave magic in the original byte order
    m_header.magic = m_data.GetU32(&offset);
    switch (m_header.magic)
    {
    case MH_MAGIC:
        m_data.SetByteOrder (eByteOrderHost);
        m_data.SetAddressByteSize(4);
        can_parse = true;
        break;

    case MH_MAGIC_64:
        m_data.SetByteOrder (eByteOrderHost);
        m_data.SetAddressByteSize(8);
        can_parse = true;
        break;

    case MH_CIGAM:
        m_data.SetByteOrder(eByteOrderHost == eByteOrderBig ? eByteOrderLittle : eByteOrderBig);
        m_data.SetAddressByteSize(4);
        can_parse = true;
        break;

    case MH_CIGAM_64:
        m_data.SetByteOrder(eByteOrderHost == eByteOrderBig ? eByteOrderLittle : eByteOrderBig);
        m_data.SetAddressByteSize(8);
        can_parse = true;
        break;

    default:
        break;
    }

    if (can_parse)
    {
        m_data.GetU32(&offset, &m_header.cputype, 6);

        ArchSpec mach_arch(m_header.cputype, m_header.cpusubtype);
        if (mach_arch == m_module->GetArchitecture())
        {
            // Read in all only the load command data
            DataBufferSP data_sp(m_file.ReadFileContents(m_offset, m_header.sizeofcmds + MachHeaderSizeFromMagic(m_header.magic)));
            m_data.SetData (data_sp);
            return true;
        }
    }
    else
    {
        memset(&m_header, 0, sizeof(struct mach_header));
    }
    return false;
}


ByteOrder
ObjectFileMachO::GetByteOrder () const
{
    lldb_private::Mutex::Locker locker(m_mutex);
    return m_data.GetByteOrder ();
}


size_t
ObjectFileMachO::GetAddressByteSize () const
{
    lldb_private::Mutex::Locker locker(m_mutex);
    return m_data.GetAddressByteSize ();
}


Symtab *
ObjectFileMachO::GetSymtab()
{
    lldb_private::Mutex::Locker locker(m_mutex);
    if (m_symtab_ap.get() == NULL)
    {
        m_symtab_ap.reset(new Symtab(this));
        ParseSymtab(false);
    }
    return m_symtab_ap.get();
}


SectionList *
ObjectFileMachO::GetSectionList()
{
    lldb_private::Mutex::Locker locker(m_mutex);
    if (m_sections_ap.get() == NULL)
    {
        m_sections_ap.reset(new SectionList());
        ParseSections();
    }
    return m_sections_ap.get();
}


size_t
ObjectFileMachO::ParseSections ()
{
    lldb::user_id_t segID = 0;
    lldb::user_id_t sectID = 0;
    struct segment_command_64 load_cmd;
    uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
    uint32_t i;
    //bool dump_sections = false;
    for (i=0; i<m_header.ncmds; ++i)
    {
        const uint32_t load_cmd_offset = offset;
        if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
            break;

        if (load_cmd.cmd == LC_SEGMENT || load_cmd.cmd == LC_SEGMENT_64)
        {
            if (m_data.GetU8(&offset, (uint8_t*)load_cmd.segname, 16))
            {
                load_cmd.vmaddr = m_data.GetAddress(&offset);
                load_cmd.vmsize = m_data.GetAddress(&offset);
                load_cmd.fileoff = m_data.GetAddress(&offset);
                load_cmd.filesize = m_data.GetAddress(&offset);
                if (m_data.GetU32(&offset, &load_cmd.maxprot, 4))
                {
                    // Keep a list of mach segments around in case we need to
                    // get at data that isn't stored in the abstracted Sections.
                    m_mach_segments.push_back (load_cmd);

                    ConstString segment_name (load_cmd.segname, std::min<int>(strlen(load_cmd.segname), sizeof(load_cmd.segname)));
                    // Use a segment ID of the segment index shifted left by 8 so they
                    // never conflict with any of the sections.
                    SectionSP segment_sp;
                    if (segment_name)
                    {
                        segment_sp.reset(new Section (NULL,
                                                      GetModule(),            // Module to which this section belongs
                                                      ++segID << 8,           // Section ID is the 1 based segment index shifted right by 8 bits as not to collide with any of the 256 section IDs that are possible
                                                      segment_name,           // Name of this section
                                                      eSectionTypeContainer,  // This section is a container of other sections.
                                                      load_cmd.vmaddr,        // File VM address == addresses as they are found in the object file
                                                      load_cmd.vmsize,        // VM size in bytes of this section
                                                      load_cmd.fileoff,       // Offset to the data for this section in the file
                                                      load_cmd.filesize,      // Size in bytes of this section as found in the the file
                                                      load_cmd.flags));       // Flags for this section

                        m_sections_ap->AddSection(segment_sp);
                    }

                    struct section_64 sect64;
                    ::bzero (&sect64, sizeof(sect64));
                    // Push a section into our mach sections for the section at
                    // index zero (NO_SECT)
                    m_mach_sections.push_back(sect64);
                    uint32_t segment_sect_idx;
                    const lldb::user_id_t first_segment_sectID = sectID + 1;


                    const uint32_t num_u32s = load_cmd.cmd == LC_SEGMENT ? 7 : 8;
                    for (segment_sect_idx=0; segment_sect_idx<load_cmd.nsects; ++segment_sect_idx)
                    {
                        if (m_data.GetU8(&offset, (uint8_t*)sect64.sectname, sizeof(sect64.sectname)) == NULL)
                            break;
                        if (m_data.GetU8(&offset, (uint8_t*)sect64.segname, sizeof(sect64.segname)) == NULL)
                            break;
                        sect64.addr = m_data.GetAddress(&offset);
                        sect64.size = m_data.GetAddress(&offset);

                        if (m_data.GetU32(&offset, &sect64.offset, num_u32s) == NULL)
                            break;

                        // Keep a list of mach sections around in case we need to
                        // get at data that isn't stored in the abstracted Sections.
                        m_mach_sections.push_back (sect64);

                        ConstString section_name (sect64.sectname, std::min<size_t>(strlen(sect64.sectname), sizeof(sect64.sectname)));
                        if (!segment_name)
                        {
                            // We have a segment with no name so we need to conjure up
                            // segments that correspond to the section's segname if there
                            // isn't already such a section. If there is such a section,
                            // we resize the section so that it spans all sections.
                            // We also mark these sections as fake so address matches don't
                            // hit if they land in the gaps between the child sections.
                            segment_name.SetTrimmedCStringWithLength(sect64.segname, sizeof(sect64.segname));
                            segment_sp = m_sections_ap->FindSectionByName (segment_name);
                            if (segment_sp.get())
                            {
                                Section *segment = segment_sp.get();
                                // Grow the section size as needed.
                                const lldb::addr_t sect64_min_addr = sect64.addr;
                                const lldb::addr_t sect64_max_addr = sect64_min_addr + sect64.size;
                                const lldb::addr_t curr_seg_byte_size = segment->GetByteSize();
                                const lldb::addr_t curr_seg_min_addr = segment->GetFileAddress();
                                const lldb::addr_t curr_seg_max_addr = curr_seg_min_addr + curr_seg_byte_size;
                                if (sect64_min_addr >= curr_seg_min_addr)
                                {
                                    const lldb::addr_t new_seg_byte_size = sect64_max_addr - curr_seg_min_addr;
                                    // Only grow the section size if needed
                                    if (new_seg_byte_size > curr_seg_byte_size)
                                        segment->SetByteSize (new_seg_byte_size);
                                }
                                else
                                {
                                    // We need to change the base address of the segment and
                                    // adjust the child section offsets for all existing children.
                                    const lldb::addr_t slide_amount = sect64_min_addr - curr_seg_min_addr;
                                    segment->Slide(slide_amount, false);
                                    segment->GetChildren().Slide (-slide_amount, false);
                                    segment->SetByteSize (curr_seg_max_addr - sect64_min_addr);
                                }
                            }
                            else
                            {
                                // Create a fake section for the section's named segment
                                segment_sp.reset(new Section(segment_sp.get(),       // Parent section
                                                             GetModule(),            // Module to which this section belongs
                                                             ++segID << 8,           // Section ID is the 1 based segment index shifted right by 8 bits as not to collide with any of the 256 section IDs that are possible
                                                             segment_name,           // Name of this section
                                                             eSectionTypeContainer,  // This section is a container of other sections.
                                                             sect64.addr,            // File VM address == addresses as they are found in the object file
                                                             sect64.size,            // VM size in bytes of this section
                                                             sect64.offset,          // Offset to the data for this section in the file
                                                             sect64.offset ? sect64.size : 0,        // Size in bytes of this section as found in the the file
                                                             load_cmd.flags));       // Flags for this section
                                segment_sp->SetIsFake(true);
                                m_sections_ap->AddSection(segment_sp);
                            }
                        }
                        assert (segment_sp.get());

                        uint32_t mach_sect_type = sect64.flags & SECTION_TYPE;
                        static ConstString g_sect_name_objc_data ("__objc_data");
                        static ConstString g_sect_name_objc_msgrefs ("__objc_msgrefs");
                        static ConstString g_sect_name_objc_selrefs ("__objc_selrefs");
                        static ConstString g_sect_name_objc_classrefs ("__objc_classrefs");
                        static ConstString g_sect_name_objc_superrefs ("__objc_superrefs");
                        static ConstString g_sect_name_objc_const ("__objc_const");
                        static ConstString g_sect_name_objc_classlist ("__objc_classlist");
                        static ConstString g_sect_name_cfstring ("__cfstring");
                        SectionType sect_type = eSectionTypeOther;

                        if (section_name == g_sect_name_objc_selrefs)
                        {
                            sect_type = eSectionTypeDataCStringPointers;
                        }
                        else if (section_name == g_sect_name_objc_msgrefs)
                        {
                            sect_type = eSectionTypeDataObjCMessageRefs;
                        }
                        else if (section_name == g_sect_name_objc_data ||
                                 section_name == g_sect_name_objc_classrefs ||
                                 section_name == g_sect_name_objc_superrefs ||
                                 section_name == g_sect_name_objc_const ||
                                 section_name == g_sect_name_objc_classlist)
                        {
                            sect_type = eSectionTypeDataPointers;
                        }
                        else if (section_name == g_sect_name_cfstring)
                        {
                            sect_type = eSectionTypeDataObjCCFStrings;
                        }

                        if (sect_type == eSectionTypeOther)
                        {
                            switch (mach_sect_type)
                            {
                            // TODO: categorize sections by other flags for regular sections
                            case S_REGULAR:

                                sect_type = eSectionTypeOther;
                                break;
                            case S_ZEROFILL:                    sect_type = eSectionTypeZeroFill; break;
                            case S_CSTRING_LITERALS:            sect_type = eSectionTypeDataCString;    break; // section with only literal C strings
                            case S_4BYTE_LITERALS:              sect_type = eSectionTypeData4;    break; // section with only 4 byte literals
                            case S_8BYTE_LITERALS:              sect_type = eSectionTypeData8;    break; // section with only 8 byte literals
                            case S_LITERAL_POINTERS:            sect_type = eSectionTypeDataPointers;  break; // section with only pointers to literals
                            case S_NON_LAZY_SYMBOL_POINTERS:    sect_type = eSectionTypeDataPointers;  break; // section with only non-lazy symbol pointers
                            case S_LAZY_SYMBOL_POINTERS:        sect_type = eSectionTypeDataPointers;  break; // section with only lazy symbol pointers
                            case S_SYMBOL_STUBS:                sect_type = eSectionTypeCode;  break; // section with only symbol stubs, byte size of stub in the reserved2 field
                            case S_MOD_INIT_FUNC_POINTERS:      sect_type = eSectionTypeDataPointers;    break; // section with only function pointers for initialization
                            case S_MOD_TERM_FUNC_POINTERS:      sect_type = eSectionTypeDataPointers; break; // section with only function pointers for termination
                            case S_COALESCED:                   sect_type = eSectionTypeOther; break;
                            case S_GB_ZEROFILL:                 sect_type = eSectionTypeZeroFill; break;
                            case S_INTERPOSING:                 sect_type = eSectionTypeCode;  break; // section with only pairs of function pointers for interposing
                            case S_16BYTE_LITERALS:             sect_type = eSectionTypeData16; break; // section with only 16 byte literals
                            case S_DTRACE_DOF:                  sect_type = eSectionTypeDebug; break;
                            case S_LAZY_DYLIB_SYMBOL_POINTERS:  sect_type = eSectionTypeDataPointers;  break;
                            default: break;
                            }
                        }

                        SectionSP section_sp(new Section(segment_sp.get(),
                                                         GetModule(),
                                                         ++sectID,
                                                         section_name,
                                                         sect_type,
                                                         sect64.addr - segment_sp->GetFileAddress(),
                                                         sect64.size,
                                                         sect64.offset,
                                                         sect64.offset == 0 ? 0 : sect64.size,
                                                         sect64.flags));
                        segment_sp->GetChildren().AddSection(section_sp);

                        if (segment_sp->IsFake())
                        {
                            segment_sp.reset();
                            segment_name.Clear();
                        }
                    }
                    if (m_header.filetype == MH_DSYM)
                    {
                        if (first_segment_sectID <= sectID)
                        {
                            lldb::user_id_t sect_uid;
                            for (sect_uid = first_segment_sectID; sect_uid <= sectID; ++sect_uid)
                            {
                                SectionSP curr_section_sp(segment_sp->GetChildren().FindSectionByID (sect_uid));
                                SectionSP next_section_sp;
                                if (sect_uid + 1 <= sectID)
                                    next_section_sp = segment_sp->GetChildren().FindSectionByID (sect_uid+1);

                                if (curr_section_sp.get())
                                {
                                    if (curr_section_sp->GetByteSize() == 0)
                                    {
                                        if (next_section_sp.get() != NULL)
                                            curr_section_sp->SetByteSize ( next_section_sp->GetFileAddress() - curr_section_sp->GetFileAddress() );
                                        else
                                            curr_section_sp->SetByteSize ( load_cmd.vmsize );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (load_cmd.cmd == LC_DYSYMTAB)
        {
            m_dysymtab.cmd = load_cmd.cmd;
            m_dysymtab.cmdsize = load_cmd.cmdsize;
            m_data.GetU32 (&offset, &m_dysymtab.ilocalsym, (sizeof(m_dysymtab) / sizeof(uint32_t)) - 2);
        }

        offset = load_cmd_offset + load_cmd.cmdsize;
    }
//    if (dump_sections)
//    {
//        StreamFile s(stdout);
//        m_sections_ap->Dump(&s, true);
//    }
    return sectID;  // Return the number of sections we registered with the module
}

class MachSymtabSectionInfo
{
public:

    MachSymtabSectionInfo (SectionList *section_list) :
        m_section_list (section_list),
        m_section_infos()
    {
        // Get the number of sections down to a depth of 1 to include
        // all segments and their sections, but no other sections that
        // may be added for debug map or
        m_section_infos.resize(section_list->GetNumSections(1));
    }


    Section *
    GetSection (uint8_t n_sect, addr_t file_addr)
    {
        if (n_sect == 0)
            return NULL;
        if (n_sect < m_section_infos.size())
        {
            if (m_section_infos[n_sect].section == NULL)
            {
                Section *section = m_section_list->FindSectionByID (n_sect).get();
                m_section_infos[n_sect].section = section;
                assert (section != NULL);
                m_section_infos[n_sect].vm_range.SetBaseAddress (section->GetFileAddress());
                m_section_infos[n_sect].vm_range.SetByteSize (section->GetByteSize());
            }
            if (m_section_infos[n_sect].vm_range.Contains(file_addr))
                return m_section_infos[n_sect].section;
        }
        return m_section_list->FindSectionContainingFileAddress(file_addr).get();
    }

protected:
    struct SectionInfo
    {
        SectionInfo () :
            vm_range(),
            section (NULL)
        {
        }

        VMRange vm_range;
        Section *section;
    };
    SectionList *m_section_list;
    std::vector<SectionInfo> m_section_infos;
};



size_t
ObjectFileMachO::ParseSymtab (bool minimize)
{
    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "ObjectFileMachO::ParseSymtab () module = %s",
                       m_file.GetFilename().AsCString(""));
    struct symtab_command symtab_load_command;
    uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
    uint32_t i;
    for (i=0; i<m_header.ncmds; ++i)
    {
        const uint32_t cmd_offset = offset;
        // Read in the load command and load command size
        if (m_data.GetU32(&offset, &symtab_load_command, 2) == NULL)
            break;
        // Watch for the symbol table load command
        if (symtab_load_command.cmd == LC_SYMTAB)
        {
            // Read in the rest of the symtab load command
            if (m_data.GetU32(&offset, &symtab_load_command.symoff, 4))
            {
                Symtab *symtab = m_symtab_ap.get();
                SectionList *section_list = GetSectionList();
                assert(section_list);
                const size_t addr_size = m_data.GetAddressByteSize();
                const ByteOrder endian = m_data.GetByteOrder();
                bool bit_width_32 = addr_size == 4;
                const size_t nlist_size = bit_width_32 ? sizeof(struct nlist) : sizeof(struct nlist_64);

                DataBufferSP symtab_data_sp(m_file.ReadFileContents(m_offset + symtab_load_command.symoff, symtab_load_command.nsyms * nlist_size));
                DataBufferSP strtab_data_sp(m_file.ReadFileContents(m_offset + symtab_load_command.stroff, symtab_load_command.strsize));

                const char *strtab_data = (const char *)strtab_data_sp->GetBytes();
//                DataExtractor symtab_data(symtab_data_sp, endian, addr_size);
//                DataExtractor strtab_data(strtab_data_sp, endian, addr_size);

                static ConstString g_segment_name_TEXT ("__TEXT");
                static ConstString g_segment_name_DATA ("__DATA");
                static ConstString g_segment_name_OBJC ("__OBJC");
                static ConstString g_section_name_eh_frame ("__eh_frame");
                SectionSP text_section_sp(section_list->FindSectionByName(g_segment_name_TEXT));
                SectionSP data_section_sp(section_list->FindSectionByName(g_segment_name_DATA));
                SectionSP objc_section_sp(section_list->FindSectionByName(g_segment_name_OBJC));
                SectionSP eh_frame_section_sp;
                if (text_section_sp.get())
                    eh_frame_section_sp = text_section_sp->GetChildren().FindSectionByName (g_section_name_eh_frame);
                else
                    eh_frame_section_sp = section_list->FindSectionByName (g_section_name_eh_frame);

                uint8_t TEXT_eh_frame_sectID = eh_frame_section_sp.get() ? eh_frame_section_sp->GetID() : NO_SECT;
                //uint32_t symtab_offset = 0;
                const uint8_t* nlist_data = symtab_data_sp->GetBytes();
                assert (symtab_data_sp->GetByteSize()/nlist_size >= symtab_load_command.nsyms);


                if (endian != eByteOrderHost)
                {
                    // ...
                    assert (!"UNIMPLEMENTED: Swap all nlist entries");
                }
                uint32_t N_SO_index = UINT_MAX;

                MachSymtabSectionInfo section_info (section_list);
                std::vector<uint32_t> N_FUN_indexes;
                std::vector<uint32_t> N_NSYM_indexes;
                std::vector<uint32_t> N_INCL_indexes;
                std::vector<uint32_t> N_BRAC_indexes;
                std::vector<uint32_t> N_COMM_indexes;
                uint32_t nlist_idx = 0;
                Symbol *symbol_ptr = NULL;

                uint32_t sym_idx = 0;
                Symbol *sym = symtab->Resize (symtab_load_command.nsyms + m_dysymtab.nindirectsyms);
                uint32_t num_syms = symtab->GetNumSymbols();

                //symtab->Reserve (symtab_load_command.nsyms + m_dysymtab.nindirectsyms);
                for (nlist_idx = 0; nlist_idx < symtab_load_command.nsyms; ++nlist_idx)
                {
                    struct nlist_64 nlist;
                    if (bit_width_32)
                    {
                        struct nlist* nlist32_ptr = (struct nlist*)(nlist_data + (nlist_idx * nlist_size));
                        nlist.n_un.n_strx = nlist32_ptr->n_un.n_strx;
                        nlist.n_type = nlist32_ptr->n_type;
                        nlist.n_sect = nlist32_ptr->n_sect;
                        nlist.n_desc = nlist32_ptr->n_desc;
                        nlist.n_value = nlist32_ptr->n_value;
                    }
                    else
                    {
                        nlist = *((struct nlist_64*)(nlist_data + (nlist_idx * nlist_size)));
                    }

                    SymbolType type = eSymbolTypeInvalid;
                    const char* symbol_name = &strtab_data[nlist.n_un.n_strx];
                    if (symbol_name[0] == '\0')
                        symbol_name = NULL;
                    Section* symbol_section = NULL;
                    bool add_nlist = true;
                    bool is_debug = ((nlist.n_type & N_STAB) != 0);

                    assert (sym_idx < num_syms);

                    sym[sym_idx].SetDebug (is_debug);

                    if (is_debug)
                    {
                        switch (nlist.n_type)
                        {
                        case N_GSYM:    // global symbol: name,,NO_SECT,type,0
                            // Sometimes the N_GSYM value contains the address.
                            if (nlist.n_value != 0)
                                symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                            type = eSymbolTypeGlobal;
                            break;

                        case N_FNAME:   // procedure name (f77 kludge): name,,NO_SECT,0,0
                            type = eSymbolTypeFunction;
                            break;

                        case N_FUN:                                         // procedure: name,,n_sect,linenumber,address
                            if (symbol_name)
                            {
                                type = eSymbolTypeFunction;
                                symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                // We use the current number of symbols in the symbol table in lieu of
                                // using nlist_idx in case we ever start trimming entries out
                                N_FUN_indexes.push_back(sym_idx);
                            }
                            else
                            {
                                type = eSymbolTypeFunctionEnd;

                                if ( !N_FUN_indexes.empty() )
                                {
                                    // Copy the size of the function into the original STAB entry so we don't have
                                    // to hunt for it later
                                    symtab->SymbolAtIndex(N_FUN_indexes.back())->SetByteSize(nlist.n_value);
                                    N_FUN_indexes.pop_back();
                                    // We dont' really need the end function STAB as it contains the size which
                                    // we already placed with the original symbol, so don't add it if we want a
                                    // minimal symbol table
                                    if (minimize)
                                        add_nlist = false;
                                }
                            }
                            break;

                        case N_STSYM:   // static symbol: name,,n_sect,type,address
                            symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                            type = eSymbolTypeStatic;
                            break;

                        case N_LCSYM:   // .lcomm symbol: name,,n_sect,type,address
                            symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                            type = eSymbolTypeCommonBlock;
                            break;

                        case N_BNSYM:
                            // We use the current number of symbols in the symbol table in lieu of
                            // using nlist_idx in case we ever start trimming entries out
                            if (minimize)
                            {
                                // Skip these if we want minimal symbol tables
                                add_nlist = false;
                            }
                            else
                            {
                                symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                N_NSYM_indexes.push_back(sym_idx);
                                type = eSymbolTypeScopeBegin;
                            }
                            break;

                        case N_ENSYM:
                            // Set the size of the N_BNSYM to the terminating index of this N_ENSYM
                            // so that we can always skip the entire symbol if we need to navigate
                            // more quickly at the source level when parsing STABS
                            if (minimize)
                            {
                                // Skip these if we want minimal symbol tables
                                add_nlist = false;
                            }
                            else
                            {
                                if ( !N_NSYM_indexes.empty() )
                                {
                                    symbol_ptr = symtab->SymbolAtIndex(N_NSYM_indexes.back());
                                    symbol_ptr->SetByteSize(sym_idx + 1);
                                    symbol_ptr->SetSizeIsSibling(true);
                                    N_NSYM_indexes.pop_back();
                                }
                                type = eSymbolTypeScopeEnd;
                            }
                            break;


                        case N_OPT:     // emitted with gcc2_compiled and in gcc source
                            type = eSymbolTypeCompiler;
                            break;

                        case N_RSYM:    // register sym: name,,NO_SECT,type,register
                            type = eSymbolTypeVariable;
                            break;

                        case N_SLINE:   // src line: 0,,n_sect,linenumber,address
                            symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                            type = eSymbolTypeLineEntry;
                            break;

                        case N_SSYM:    // structure elt: name,,NO_SECT,type,struct_offset
                            type = eSymbolTypeVariableType;
                            break;

                        case N_SO:
                            type = eSymbolTypeSourceFile;
                            if (symbol_name == NULL)
                            {
                                if (N_SO_index == UINT_MAX)
                                {
                                    // Skip the extra blank N_SO entries that happen when the entire
                                    // path is contained in the second consecutive N_SO STAB.
                                    if (minimize)
                                        add_nlist = false;
                                }
                                else
                                {
                                    // Set the size of the N_SO to the terminating index of this N_SO
                                    // so that we can always skip the entire N_SO if we need to navigate
                                    // more quickly at the source level when parsing STABS
                                    symbol_ptr = symtab->SymbolAtIndex(N_SO_index);
                                    symbol_ptr->SetByteSize(sym_idx + 1);
                                    symbol_ptr->SetSizeIsSibling(true);
                                }
                                N_NSYM_indexes.clear();
                                N_INCL_indexes.clear();
                                N_BRAC_indexes.clear();
                                N_COMM_indexes.clear();
                                N_FUN_indexes.clear();
                                N_SO_index = UINT_MAX;
                            }
                            else if (symbol_name[0] == '/')
                            {
                                // We use the current number of symbols in the symbol table in lieu of
                                // using nlist_idx in case we ever start trimming entries out
                                N_SO_index = sym_idx;
                            }
                            break;

                        case N_OSO:     // object file name: name,,0,0,st_mtime
                            type = eSymbolTypeObjectFile;
                            break;

                        case N_LSYM:    // local sym: name,,NO_SECT,type,offset
                            type = eSymbolTypeLocal;
                            break;

                        //----------------------------------------------------------------------
                        // INCL scopes
                        //----------------------------------------------------------------------
                        case N_BINCL:   // include file beginning: name,,NO_SECT,0,sum
                            // We use the current number of symbols in the symbol table in lieu of
                            // using nlist_idx in case we ever start trimming entries out
                            N_INCL_indexes.push_back(sym_idx);
                            type = eSymbolTypeScopeBegin;
                            break;
                        case N_EINCL:   // include file end: name,,NO_SECT,0,0

                            // Set the size of the N_BINCL to the terminating index of this N_EINCL
                            // so that we can always skip the entire symbol if we need to navigate
                            // more quickly at the source level when parsing STABS
                            if ( !N_INCL_indexes.empty() )
                            {
                                symbol_ptr = symtab->SymbolAtIndex(N_INCL_indexes.back());
                                symbol_ptr->SetByteSize(sym_idx + 1);
                                symbol_ptr->SetSizeIsSibling(true);
                                N_INCL_indexes.pop_back();
                            }
                            type = eSymbolTypeScopeEnd;
                            break;

                        case N_SOL:     // #included file name: name,,n_sect,0,address
                            type = eSymbolTypeHeaderFile;
                            break;

                        case N_PARAMS:  // compiler parameters: name,,NO_SECT,0,0
                            type = eSymbolTypeCompiler;
                            break;

                        case N_VERSION: // compiler version: name,,NO_SECT,0,0
                            type = eSymbolTypeCompiler;
                            break;

                        case N_OLEVEL:  // compiler -O level: name,,NO_SECT,0,0
                            type = eSymbolTypeCompiler;
                            break;

                        case N_PSYM:    // parameter: name,,NO_SECT,type,offset
                            type = eSymbolTypeVariable;
                            break;

                        case N_ENTRY:   // alternate entry: name,,n_sect,linenumber,address
                            symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                            type = eSymbolTypeLineEntry;
                            break;

                        //----------------------------------------------------------------------
                        // Left and Right Braces
                        //----------------------------------------------------------------------
                        case N_LBRAC:   // left bracket: 0,,NO_SECT,nesting level,address
                            // We use the current number of symbols in the symbol table in lieu of
                            // using nlist_idx in case we ever start trimming entries out
                            symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                            N_BRAC_indexes.push_back(sym_idx);
                            type = eSymbolTypeScopeBegin;
                            break;

                        case N_RBRAC:   // right bracket: 0,,NO_SECT,nesting level,address
                            // Set the size of the N_LBRAC to the terminating index of this N_RBRAC
                            // so that we can always skip the entire symbol if we need to navigate
                            // more quickly at the source level when parsing STABS
                            symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                            if ( !N_BRAC_indexes.empty() )
                            {
                                symbol_ptr = symtab->SymbolAtIndex(N_BRAC_indexes.back());
                                symbol_ptr->SetByteSize(sym_idx + 1);
                                symbol_ptr->SetSizeIsSibling(true);
                                N_BRAC_indexes.pop_back();
                            }
                            type = eSymbolTypeScopeEnd;
                            break;

                        case N_EXCL:    // deleted include file: name,,NO_SECT,0,sum
                            type = eSymbolTypeHeaderFile;
                            break;

                        //----------------------------------------------------------------------
                        // COMM scopes
                        //----------------------------------------------------------------------
                        case N_BCOMM:   // begin common: name,,NO_SECT,0,0
                            // We use the current number of symbols in the symbol table in lieu of
                            // using nlist_idx in case we ever start trimming entries out
                            type = eSymbolTypeScopeBegin;
                            N_COMM_indexes.push_back(sym_idx);
                            break;

                        case N_ECOML:   // end common (local name): 0,,n_sect,0,address
                            symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                            // Fall through

                        case N_ECOMM:   // end common: name,,n_sect,0,0
                            // Set the size of the N_BCOMM to the terminating index of this N_ECOMM/N_ECOML
                            // so that we can always skip the entire symbol if we need to navigate
                            // more quickly at the source level when parsing STABS
                            if ( !N_COMM_indexes.empty() )
                            {
                                symbol_ptr = symtab->SymbolAtIndex(N_COMM_indexes.back());
                                symbol_ptr->SetByteSize(sym_idx + 1);
                                symbol_ptr->SetSizeIsSibling(true);
                                N_COMM_indexes.pop_back();
                            }
                            type = eSymbolTypeScopeEnd;
                            break;

                        case N_LENG:    // second stab entry with length information
                            type = eSymbolTypeAdditional;
                            break;

                        default: break;
                        }
                    }
                    else
                    {
                        //uint8_t n_pext    = N_PEXT & nlist.n_type;
                        uint8_t n_type  = N_TYPE & nlist.n_type;
                        sym[sym_idx].SetExternal((N_EXT & nlist.n_type) != 0);

                        if (symbol_name && ::strstr (symbol_name, ".objc") == symbol_name)
                        {
                            type = eSymbolTypeRuntime;
                        }
                        else
                        {
                            switch (n_type)
                            {
                            case N_INDR:    // Fall through
                            case N_PBUD:    // Fall through
                            case N_UNDF:
                                type = eSymbolTypeExtern;
                                break;

                            case N_ABS:
                                type = eSymbolTypeAbsolute;
                                break;

                            case N_SECT:
                                symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);

                                assert(symbol_section != NULL);
                                if (TEXT_eh_frame_sectID == nlist.n_sect)
                                {
                                    type = eSymbolTypeException;
                                }
                                else
                                {
                                    uint32_t section_type = symbol_section->GetAllFlagBits() & SECTION_TYPE;

                                    switch (section_type)
                                    {
                                    case S_REGULAR:                     break; // regular section
                                    //case S_ZEROFILL:                  type = eSymbolTypeData;    break; // zero fill on demand section
                                    case S_CSTRING_LITERALS:            type = eSymbolTypeData;    break; // section with only literal C strings
                                    case S_4BYTE_LITERALS:              type = eSymbolTypeData;    break; // section with only 4 byte literals
                                    case S_8BYTE_LITERALS:              type = eSymbolTypeData;    break; // section with only 8 byte literals
                                    case S_LITERAL_POINTERS:            type = eSymbolTypeTrampoline; break; // section with only pointers to literals
                                    case S_NON_LAZY_SYMBOL_POINTERS:    type = eSymbolTypeTrampoline; break; // section with only non-lazy symbol pointers
                                    case S_LAZY_SYMBOL_POINTERS:        type = eSymbolTypeTrampoline; break; // section with only lazy symbol pointers
                                    case S_SYMBOL_STUBS:                type = eSymbolTypeTrampoline; break; // section with only symbol stubs, byte size of stub in the reserved2 field
                                    case S_MOD_INIT_FUNC_POINTERS:      type = eSymbolTypeCode;    break; // section with only function pointers for initialization
                                    case S_MOD_TERM_FUNC_POINTERS:      type = eSymbolTypeCode;    break; // section with only function pointers for termination
                                    //case S_COALESCED:                 type = eSymbolType;    break; // section contains symbols that are to be coalesced
                                    //case S_GB_ZEROFILL:               type = eSymbolTypeData;    break; // zero fill on demand section (that can be larger than 4 gigabytes)
                                    case S_INTERPOSING:                 type = eSymbolTypeTrampoline;  break; // section with only pairs of function pointers for interposing
                                    case S_16BYTE_LITERALS:             type = eSymbolTypeData;    break; // section with only 16 byte literals
                                    case S_DTRACE_DOF:                  type = eSymbolTypeInstrumentation; break;
                                    case S_LAZY_DYLIB_SYMBOL_POINTERS:  type = eSymbolTypeTrampoline; break;
                                    default: break;
                                    }

                                    if (type == eSymbolTypeInvalid)
                                    {
                                        const char *symbol_sect_name = symbol_section->GetName().AsCString();
                                        if (symbol_section->IsDescendant (text_section_sp.get()))
                                        {
                                            if (symbol_section->IsClear(S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SELF_MODIFYING_CODE | S_ATTR_SOME_INSTRUCTIONS))
                                                type = eSymbolTypeData;
                                            else
                                                type = eSymbolTypeCode;
                                        }
                                        else
                                        if (symbol_section->IsDescendant(data_section_sp.get()))
                                        {
                                            if (symbol_sect_name && ::strstr (symbol_sect_name, "__objc") == symbol_sect_name)
                                            {
                                                type = eSymbolTypeRuntime;
                                            }
                                            else
                                            if (symbol_sect_name && ::strstr (symbol_sect_name, "__gcc_except_tab") == symbol_sect_name)
                                            {
                                                type = eSymbolTypeException;
                                            }
                                            else
                                            {
                                                type = eSymbolTypeData;
                                            }
                                        }
                                        else
                                        if (symbol_sect_name && ::strstr (symbol_sect_name, "__IMPORT") == symbol_sect_name)
                                        {
                                            type = eSymbolTypeTrampoline;
                                        }
                                        else
                                        if (symbol_section->IsDescendant(objc_section_sp.get()))
                                        {
                                            type = eSymbolTypeRuntime;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }

                    if (add_nlist)
                    {
                        bool symbol_name_is_mangled = false;
                        if (symbol_name && symbol_name[0] == '_')
                        {
                            symbol_name_is_mangled = symbol_name[1] == '_';
                            symbol_name++;  // Skip the leading underscore
                        }
                        uint64_t symbol_value = nlist.n_value;
                        if (symbol_section != NULL)
                            symbol_value -= symbol_section->GetFileAddress();

                        sym[sym_idx].SetID (nlist_idx);
                        sym[sym_idx].SetType (type);
                        if (symbol_name)
                            sym[sym_idx].GetMangled().SetValue(symbol_name, symbol_name_is_mangled);
                        sym[sym_idx].GetAddressRangeRef().GetBaseAddress().SetSection (symbol_section);
                        sym[sym_idx].GetAddressRangeRef().GetBaseAddress().SetOffset (symbol_value);
                        sym[sym_idx].SetFlags (nlist.n_type << 16 | nlist.n_desc);

                        ++sym_idx;
                    }
                    else
                    {
                        sym[sym_idx].Clear();
                    }

                }


                // STAB N_GSYM entries end up having a symbol type eSymbolTypeGlobal and when the symbol value
                // is zero, the address of the global ends up being in a non-STAB entry. Try and fix up all
                // such entries by figuring out what the address for the global is by looking up this non-STAB
                // entry and copying the value into the debug symbol's value to save us the hassle in the
                // debug symbol parser.

                Symbol *global_symbol = NULL;
                for (nlist_idx = 0;
                     nlist_idx < symtab_load_command.nsyms && (global_symbol = symtab->FindSymbolWithType(eSymbolTypeGlobal, nlist_idx)) != NULL;
                     nlist_idx++)
                {
                    if (global_symbol->GetValue().GetFileAddress() == 0)
                    {
                        std::vector<uint32_t> indexes;
                        if (symtab->AppendSymbolIndexesWithName(global_symbol->GetMangled().GetName(), indexes) > 0)
                        {
                            std::vector<uint32_t>::const_iterator pos;
                            std::vector<uint32_t>::const_iterator end = indexes.end();
                            for (pos = indexes.begin(); pos != end; ++pos)
                            {
                                symbol_ptr = symtab->SymbolAtIndex(*pos);
                                if (symbol_ptr != global_symbol && symbol_ptr->IsDebug() == false)
                                {
                                    global_symbol->SetValue(symbol_ptr->GetValue());
                                    break;
                                }
                            }
                        }
                    }
                }
                // Now synthesize indirect symbols
                if (m_dysymtab.nindirectsyms != 0)
                {
                    DataBufferSP indirect_symbol_indexes_sp(m_file.ReadFileContents(m_offset + m_dysymtab.indirectsymoff, m_dysymtab.nindirectsyms * 4));

                    if (indirect_symbol_indexes_sp && indirect_symbol_indexes_sp->GetByteSize())
                    {
                        DataExtractor indirect_symbol_index_data (indirect_symbol_indexes_sp, m_data.GetByteOrder(), m_data.GetAddressByteSize());

                        for (uint32_t sect_idx = 1; sect_idx < m_mach_sections.size(); ++sect_idx)
                        {
                            if ((m_mach_sections[sect_idx].flags & SECTION_TYPE) == S_SYMBOL_STUBS)
                            {
                                uint32_t symbol_stub_byte_size = m_mach_sections[sect_idx].reserved2;
                                if (symbol_stub_byte_size == 0)
                                    continue;

                                const uint32_t num_symbol_stubs = m_mach_sections[sect_idx].size / symbol_stub_byte_size;

                                if (num_symbol_stubs == 0)
                                    continue;

                                const uint32_t symbol_stub_index_offset = m_mach_sections[sect_idx].reserved1;
                                uint32_t stub_sym_id = symtab_load_command.nsyms;
                                for (uint32_t stub_idx = 0; stub_idx < num_symbol_stubs; ++stub_idx)
                                {
                                    const uint32_t symbol_stub_index = symbol_stub_index_offset + stub_idx;
                                    const lldb::addr_t symbol_stub_addr = m_mach_sections[sect_idx].addr + (stub_idx * symbol_stub_byte_size);
                                    uint32_t symbol_stub_offset = symbol_stub_index * 4;
                                    if (indirect_symbol_index_data.ValidOffsetForDataOfSize(symbol_stub_offset, 4))
                                    {
                                        const uint32_t symbol_index = indirect_symbol_index_data.GetU32 (&symbol_stub_offset);

                                        Symbol *stub_symbol = symtab->SymbolAtIndex(symbol_index);
                                        if (stub_symbol)
                                        {
                                            Address so_addr(symbol_stub_addr, section_list);

                                            if (stub_symbol->GetType() == eSymbolTypeExtern)
                                            {
                                                // Change the external symbol into a trampoline that makes sense
                                                // These symbols were N_UNDF N_EXT, and are useless to us, so we
                                                // can re-use them so we don't have to make up a synthetic symbol
                                                // for no good reason.
                                                stub_symbol->SetType (eSymbolTypeTrampoline);
                                                stub_symbol->SetExternal (false);
                                                stub_symbol->GetAddressRangeRef().GetBaseAddress() = so_addr;
                                                stub_symbol->GetAddressRangeRef().SetByteSize (symbol_stub_byte_size);
                                            }
                                            else
                                            {
                                                // Make a synthetic symbol to describe the trampoline stub
                                                if (sym_idx >= num_syms)
                                                {
                                                    sym = symtab->Resize (num_syms + 16);
                                                    num_syms = symtab->GetNumSymbols();
                                                }
                                                sym[sym_idx].SetID (stub_sym_id++);
                                                sym[sym_idx].GetMangled() = stub_symbol->GetMangled();
                                                sym[sym_idx].SetType (eSymbolTypeTrampoline);
                                                sym[sym_idx].SetIsSynthetic (true);
                                                sym[sym_idx].GetAddressRangeRef().GetBaseAddress() = so_addr;
                                                sym[sym_idx].GetAddressRangeRef().SetByteSize (symbol_stub_byte_size);
                                                ++sym_idx;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (sym_idx != symtab->GetNumSymbols())
                    symtab->Resize (sym_idx);

                return symtab->GetNumSymbols();
            }
        }
        offset = cmd_offset + symtab_load_command.cmdsize;
    }
    return 0;
}


void
ObjectFileMachO::Dump (Stream *s)
{
    lldb_private::Mutex::Locker locker(m_mutex);
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    if (m_header.magic == MH_MAGIC_64 || m_header.magic == MH_CIGAM_64)
        s->PutCString("ObjectFileMachO64");
    else
        s->PutCString("ObjectFileMachO32");

    ArchSpec header_arch(m_header.cputype, m_header.cpusubtype);

    *s << ", file = '" << m_file << "', arch = " << header_arch.AsCString() << "\n";

    if (m_sections_ap.get())
        m_sections_ap->Dump(s, NULL, true);

    if (m_symtab_ap.get())
        m_symtab_ap->Dump(s, NULL);
}


bool
ObjectFileMachO::GetUUID (UUID* uuid)
{
    lldb_private::Mutex::Locker locker(m_mutex);
    struct uuid_command load_cmd;
    uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
    uint32_t i;
    for (i=0; i<m_header.ncmds; ++i)
    {
        const uint32_t cmd_offset = offset;
        if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
            break;

        if (load_cmd.cmd == LC_UUID)
        {
            const uint8_t *uuid_bytes = m_data.PeekData(offset, 16);
            if (uuid_bytes)
            {
                uuid->SetBytes (uuid_bytes);
                return true;
            }
            return false;
        }
        offset = cmd_offset + load_cmd.cmdsize;
    }
    return false;
}


uint32_t
ObjectFileMachO::GetDependentModules (FileSpecList& files)
{
    lldb_private::Mutex::Locker locker(m_mutex);
    struct load_command load_cmd;
    uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
    uint32_t count = 0;
    uint32_t i;
    for (i=0; i<m_header.ncmds; ++i)
    {
        const uint32_t cmd_offset = offset;
        if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
            break;

        switch (load_cmd.cmd)
        {
        case LC_LOAD_DYLIB:
        case LC_LOAD_WEAK_DYLIB:
        case LC_REEXPORT_DYLIB:
        case LC_LOAD_DYLINKER:
        case LC_LOADFVMLIB:
            {
                uint32_t name_offset = cmd_offset + m_data.GetU32(&offset);
                const char *path = m_data.PeekCStr(name_offset);
                // Skip any path that starts with '@' since these are usually:
                // @executable_path/.../file
                // @rpath/.../file
                if (path && path[0] != '@')
                {
                    FileSpec file_spec(path);
                    if (files.AppendIfUnique(file_spec))
                        count++;
                }
            }
            break;

        default:
            break;
        }
        offset = cmd_offset + load_cmd.cmdsize;
    }
    return count;
}

bool
ObjectFileMachO::GetTargetTriple (ConstString &target_triple)
{
    lldb_private::Mutex::Locker locker(m_mutex);
    std::string triple(GetModule()->GetArchitecture().AsCString());
    triple += "-apple-darwin";
    target_triple.SetCString(triple.c_str());
    if (target_triple)
        return true;
    return false;
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ObjectFileMachO::GetPluginName()
{
    return "ObjectFileMachO";
}

const char *
ObjectFileMachO::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjectFileMachO::GetPluginVersion()
{
    return 1;
}

void
ObjectFileMachO::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
ObjectFileMachO::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
ObjectFileMachO::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}




