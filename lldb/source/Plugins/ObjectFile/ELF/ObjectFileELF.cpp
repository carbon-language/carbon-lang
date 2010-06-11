//===-- ObjectFileELF.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjectFileELF.h"

#include <mach/machine.h>
#include <assert.h>

#include <algorithm>

#include <stdint.h>
#include "elf.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/ObjectFile.h"

#define CASE_AND_STREAM(s, def, width)  case def: s->Printf("%-*s", width, #def); break;

using namespace lldb;
using namespace lldb_private;
using namespace std;


#include <mach-o/nlist.h>
#include <mach-o/stab.h>


void
ObjectFileELF::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
ObjectFileELF::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
ObjectFileELF::GetPluginNameStatic()
{
    return "object-file.elf32";
}

const char *
ObjectFileELF::GetPluginDescriptionStatic()
{
    return "ELF object file reader (32-bit).";
}


ObjectFile *
ObjectFileELF::CreateInstance (Module* module, DataBufferSP& dataSP, const FileSpec* file, addr_t offset, addr_t length)
{
    if (ObjectFileELF::MagicBytesMatch(dataSP))
    {
        std::auto_ptr<ObjectFile> objfile_ap(new ObjectFileELF (module, dataSP, file, offset, length));
        if (objfile_ap.get() && objfile_ap->ParseHeader())
            return objfile_ap.release();
    }
    return NULL;
}

bool
ObjectFileELF::MagicBytesMatch (DataBufferSP& dataSP)
{
    DataExtractor data(dataSP, eByteOrderHost, 4);
    const uint8_t* magic = data.PeekData(0, 4);
    if (magic != NULL)
    {
        return magic[EI_MAG0] == 0x7f
            && magic[EI_MAG1] == 'E'
            && magic[EI_MAG2] == 'L'
            && magic[EI_MAG3] == 'F';
    }
    return false;
}


ObjectFileELF::ObjectFileELF(Module* module, DataBufferSP& dataSP, const FileSpec* file, addr_t offset, addr_t length) :
    ObjectFile (module, file, offset, length, dataSP),
    m_header(),
    m_program_headers(),
    m_section_headers(),
    m_sections_ap(),
    m_symtab_ap(),
    m_shstr_data()
{
    if (file)
        m_file = *file;
    ::bzero (&m_header, sizeof(m_header));
}


ObjectFileELF::~ObjectFileELF()
{
}

ByteOrder
ObjectFileELF::GetByteOrder () const
{
    if (m_header.e_ident[EI_DATA] == ELFDATA2MSB)
        return eByteOrderBig;
    if (m_header.e_ident[EI_DATA] == ELFDATA2LSB)
        return eByteOrderLittle;
    return eByteOrderInvalid;
}

size_t
ObjectFileELF::GetAddressByteSize () const
{
    return m_data.GetAddressByteSize();
}

bool
ObjectFileELF::ParseHeader ()
{
    m_data.SetAddressByteSize(4);
    uint32_t offset = GetOffset();
    if (m_data.GetU8(&offset, m_header.e_ident, EI_NIDENT) == NULL)
        return false;

    m_data.SetByteOrder(GetByteOrder());

    // Read e_type and e_machine
    if (m_data.GetU16(&offset, &m_header.e_type, 2) == NULL)
        return false;

    // read e_version, e_entry, e_phoff, e_shoff, e_flags
    if (m_data.GetU32(&offset, &m_header.e_version, 5) == NULL)
        return false;

    // Read e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx
    if (m_data.GetU16(&offset, &m_header.e_ehsize, 6) == NULL)
        return false;

    return true;
}

bool
ObjectFileELF::GetUUID (UUID* uuid)
{
    return false;
}

uint32_t
ObjectFileELF::GetDependentModules(FileSpecList& files)
{
    return 0;
}

//----------------------------------------------------------------------
// ParseProgramHeaders
//----------------------------------------------------------------------
size_t
ObjectFileELF::ParseProgramHeaders()
{
    // We have already parsed the program headers
    if (!m_program_headers.empty())
        return m_program_headers.size();

    uint32_t offset = 0;
    if (m_header.e_phnum > 0)
    {
        m_program_headers.resize(m_header.e_phnum);

        if (m_program_headers.size() != m_header.e_phnum)
            return 0;

        const size_t byte_size = m_header.e_phnum * m_header.e_phentsize;
        DataBufferSP buffer_sp(m_file.ReadFileContents(m_offset + m_header.e_phoff, byte_size));

        if (buffer_sp.get() == NULL || buffer_sp->GetByteSize() != byte_size)
            return 0;

        DataExtractor data(buffer_sp, m_data.GetByteOrder(), m_data.GetAddressByteSize());

        uint32_t idx;
        for (idx = 0; idx < m_header.e_phnum; ++idx)
        {
            if (data.GetU32(&offset, &m_program_headers[idx].p_type, 8) == NULL)
                return 0;
        }
        if (idx < m_program_headers.size())
            m_program_headers.resize(idx);
    }

    return m_program_headers.size();
}


//----------------------------------------------------------------------
// ParseSectionHeaders
//----------------------------------------------------------------------
size_t
ObjectFileELF::ParseSectionHeaders()
{
    // We have already parsed the section headers
    if (!m_section_headers.empty())
        return m_section_headers.size();

    if (m_header.e_shnum > 0)
    {
        uint32_t offset = 0;

        m_section_headers.resize(m_header.e_shnum);

        if (m_section_headers.size() != m_header.e_shnum)
            return 0;

        const size_t byte_size = m_header.e_shnum * m_header.e_shentsize;
        DataBufferSP buffer_sp(m_file.ReadFileContents(m_offset + m_header.e_shoff, byte_size));

        if (buffer_sp.get() == NULL || buffer_sp->GetByteSize() != byte_size)
            return 0;

        DataExtractor data(buffer_sp, m_data.GetByteOrder(), m_data.GetAddressByteSize());

        uint32_t idx;
        for (idx = 0; idx < m_header.e_shnum; ++idx)
        {
            if (data.GetU32(&offset, &m_section_headers[idx].sh_name, 10) == NULL)
                break;
        }
        if (idx < m_section_headers.size())
            m_section_headers.resize(idx);
    }

    return m_section_headers.size();
}

size_t
ObjectFileELF::GetSectionHeaderStringTable()
{
    if (m_shstr_data.GetByteSize() == 0)
    {
        if (m_header.e_shstrndx && m_header.e_shstrndx < m_section_headers.size())
        {
            const size_t byte_size = m_section_headers[m_header.e_shstrndx].sh_size;
            DataBufferSP buffer_sp(m_file.ReadFileContents(m_offset + m_section_headers[m_header.e_shstrndx].sh_offset, byte_size));

            if (buffer_sp.get() == NULL || buffer_sp->GetByteSize() != byte_size)
                return 0;

            m_shstr_data.SetData(buffer_sp);
        }
    }
    return m_shstr_data.GetByteSize();
}

uint32_t
ObjectFileELF::GetSectionIndexByName(const char *name)
{
    if (ParseSectionHeaders() && GetSectionHeaderStringTable())
    {
        uint32_t offset = 1;    // Skip leading NULL string at offset 0;
        while (m_shstr_data.ValidOffset(offset))
        {
            uint32_t sh_name = offset;  // Save offset in case we find a match
            const char* sectionHeaderName = m_shstr_data.GetCStr(&offset);
            if (sectionHeaderName)
            {
                if (strcmp(name, sectionHeaderName) == 0)
                {
                    SectionHeaderCollIter pos;
                    for (pos = m_section_headers.begin(); pos != m_section_headers.end(); ++pos)
                    {
                        if ( (*pos).sh_name == sh_name )
                        {
                            // section indexes are 1 based
                            return std::distance(m_section_headers.begin(), pos) + 1;
                        }
                    }
                    return UINT32_MAX;
                }
            }
            else
            {
                return UINT32_MAX;
            }
        }
    }

    return UINT32_MAX;
}

SectionList *
ObjectFileELF::GetSectionList()
{
    if (m_sections_ap.get() == NULL)
    {
        m_sections_ap.reset(new SectionList());
        if (ParseSectionHeaders() && GetSectionHeaderStringTable())
        {
            uint32_t sh_idx = 0;
            const size_t num_sections = m_section_headers.size();
            for (sh_idx = 0; sh_idx < num_sections; ++sh_idx)
            {
                ConstString section_name(m_shstr_data.PeekCStr(m_section_headers[sh_idx].sh_name));
                uint64_t section_file_size = m_section_headers[sh_idx].sh_type == SHT_NOBITS ? 0 : m_section_headers[sh_idx].sh_size;
                SectionSP section_sp(new Section(NULL,                                  // Parent section
                                                 GetModule(),                           // Module to which this section belongs
                                                 sh_idx + 1,                            // Section ID is the 1 based
                                                 section_name,                          // Name of this section
                                                 eSectionTypeOther,  // TODO: fill this in appropriately for ELF...
                                                 m_section_headers[sh_idx].sh_addr,     // File VM address
                                                 m_section_headers[sh_idx].sh_size,     // VM size in bytes of this section
                                                 m_section_headers[sh_idx].sh_offset,   // Offset to the data for this section in the file
                                                 section_file_size,                     // Size in bytes of this section as found in the the file
                                                 m_section_headers[sh_idx].sh_flags));  // Flags for this section
                if (section_sp.get())
                    m_sections_ap->AddSection(section_sp);

            }
        }
    }
    return m_sections_ap.get();
}

static void
ParseSymbols (Symtab *symtab, SectionList *section_list, const Elf32_Shdr &symtab_shdr, const DataExtractor& symtab_data, const DataExtractor& strtab_data)
{
    assert (sizeof(Elf32_Sym) == symtab_shdr.sh_entsize);
    const uint32_t num_symbols = symtab_data.GetByteSize() / sizeof(Elf32_Sym);
    uint32_t offset = 0;
    Elf32_Sym symbol;
    uint32_t i;
    static ConstString text_section_name(".text");
    static ConstString init_section_name(".init");
    static ConstString fini_section_name(".fini");
    static ConstString ctors_section_name(".ctors");
    static ConstString dtors_section_name(".dtors");

    static ConstString data_section_name(".data");
    static ConstString rodata_section_name(".rodata");
    static ConstString rodata1_section_name(".rodata1");
    static ConstString data2_section_name(".data1");
    static ConstString bss_section_name(".bss");

    for (i=0; i<num_symbols; ++i)
    {
    //  if (symtab_data.GetU32(&offset, &symbol.st_name, 3) == 0)
    //      break;

        if (!symtab_data.ValidOffsetForDataOfSize(offset, sizeof(Elf32_Sym)))
            break;

        symbol.st_name  = symtab_data.GetU32 (&offset);
        symbol.st_value = symtab_data.GetU32 (&offset);
        symbol.st_size  = symtab_data.GetU32 (&offset);
        symbol.st_info  = symtab_data.GetU8  (&offset);
        symbol.st_other = symtab_data.GetU8  (&offset);
        symbol.st_shndx = symtab_data.GetU16 (&offset);

        Section * symbol_section = NULL;
        SymbolType symbol_type = eSymbolTypeInvalid;

        switch (symbol.st_shndx)
        {
        case SHN_ABS:
            symbol_type = eSymbolTypeAbsolute;
            break;
        case SHN_UNDEF:
            symbol_type = eSymbolTypeUndefined;
            break;
        default:
            symbol_section = section_list->GetSectionAtIndex (symbol.st_shndx).get();
            break;
        }

        switch (ELF32_ST_BIND (symbol.st_info))
        {
        default:
        case STT_NOTYPE:
            // The symbol's type is not specified.
            break;

        case STT_OBJECT:
            // The symbol is associated with a data object, such as a variable, an array, etc.
            symbol_type == eSymbolTypeData;
            break;

        case STT_FUNC:
            // The symbol is associated with a function or other executable code.
            symbol_type == eSymbolTypeCode;
            break;

        case STT_SECTION:
            // The symbol is associated with a section. Symbol table entries of
            // this type exist primarily for relocation and normally have
            // STB_LOCAL binding.
            break;

        case STT_FILE:
            // Conventionally, the symbol's name gives the name of the source
            // file associated with the object file. A file symbol has STB_LOCAL
            // binding, its section index is SHN_ABS, and it precedes the other
            // STB_LOCAL symbols for the file, if it is present.
            symbol_type == eSymbolTypeObjectFile;
            break;
        }

        if (symbol_type == eSymbolTypeInvalid)
        {
            if (symbol_section)
            {
                const ConstString &sect_name = symbol_section->GetName();
                if (sect_name == text_section_name ||
                    sect_name == init_section_name ||
                    sect_name == fini_section_name ||
                    sect_name == ctors_section_name ||
                    sect_name == dtors_section_name)
                {
                    symbol_type = eSymbolTypeCode;
                }
                else
                if (sect_name == data_section_name ||
                    sect_name == data2_section_name ||
                    sect_name == rodata_section_name ||
                    sect_name == rodata1_section_name ||
                    sect_name == bss_section_name)
                {
                    symbol_type = eSymbolTypeData;
                }
            }
        }

        uint64_t symbol_value = symbol.st_value;
        if (symbol_section != NULL)
            symbol_value -= symbol_section->GetFileAddress();
        const char *symbol_name = strtab_data.PeekCStr(symbol.st_name);

        Symbol dc_symbol(i,             // ID is the original symbol table index
                        symbol_name,    // symbol name
                        false,          // Is the symbol name mangled?
                        symbol_type,    // type of this symbol
                        ELF32_ST_BIND (symbol.st_info) == STB_GLOBAL,   // Is this globally visible?
                        false,          // Is this symbol debug info?
                        false,          // Is this symbol a trampoline?
                        false,          // Is this symbol artificial?
                        symbol_section, // section pointer if symbol_value is an offset within a section, else NULL
                        symbol_value,   // offset from section if section is non-NULL, else the value for this symbol
                        symbol.st_size, // size in bytes of this symbol
                        symbol.st_other << 8 | symbol.st_info); // symbol flags
        symtab->AddSymbol(dc_symbol);
    }
}


Symtab *
ObjectFileELF::GetSymtab()
{
    if (m_symtab_ap.get() == NULL)
    {
        m_symtab_ap.reset(new Symtab(this));

        if (ParseSectionHeaders() && GetSectionHeaderStringTable())
        {
            uint32_t symtab_idx = UINT32_MAX;
            uint32_t dynsym_idx = UINT32_MAX;
            uint32_t sh_idx = 0;
            const size_t num_sections = m_section_headers.size();
            for (sh_idx = 0; sh_idx < num_sections; ++sh_idx)
            {
                if (m_section_headers[sh_idx].sh_type == SHT_SYMTAB)
                {
                    symtab_idx = sh_idx;
                    break;
                }
                if (m_section_headers[sh_idx].sh_type == SHT_DYNSYM)
                {
                    dynsym_idx = sh_idx;
                }
            }

            SectionList *section_list = NULL;
            static ConstString g_symtab(".symtab");
            static ConstString g_strtab(".strtab");
            static ConstString g_dynsym(".dynsym");
            static ConstString g_dynstr(".dynstr");
            // Check if we found a full symbol table?
            if (symtab_idx < num_sections)
            {
                section_list = GetSectionList();
                if (section_list)
                {
                    Section *symtab_section = section_list->FindSectionByName(g_symtab).get();
                    Section *strtab_section = section_list->FindSectionByName(g_strtab).get();
                    if (symtab_section && strtab_section)
                    {
                        DataExtractor symtab_data;
                        DataExtractor strtab_data;
                        if (symtab_section->ReadSectionDataFromObjectFile (this, symtab_data) > 0 &&
                            strtab_section->ReadSectionDataFromObjectFile (this, strtab_data) > 0)
                        {
                            ParseSymbols (m_symtab_ap.get(), section_list, m_section_headers[symtab_idx], symtab_data, strtab_data);
                        }
                    }
                }
            }
            // Check if we found a reduced symbol table that gets used for dynamic linking?
            else if (dynsym_idx < num_sections)
            {
                section_list = GetSectionList();
                if (section_list)
                {
                    Section *dynsym_section = section_list->FindSectionByName(g_dynsym).get();
                    Section *dynstr_section = section_list->FindSectionByName(g_dynstr).get();
                    if (dynsym_section && dynstr_section)
                    {
                        DataExtractor dynsym_data;
                        DataExtractor dynstr_data;
                        if (dynsym_section->ReadSectionDataFromObjectFile (this, dynsym_data) > 0 &&
                            dynstr_section->ReadSectionDataFromObjectFile (this, dynstr_data) > 0)
                        {
                            ParseSymbols (m_symtab_ap.get(), section_list, m_section_headers[dynsym_idx], dynsym_data, dynstr_data);
                        }
                    }
                }
            }
        }
    }
    return m_symtab_ap.get();
}

//
////----------------------------------------------------------------------
//// GetNListSymtab
////----------------------------------------------------------------------
//bool
//ELF32RuntimeFileParser::GetNListSymtab(BinaryDataRef& stabs_data, BinaryDataRef& stabstr_data, bool locals_only, uint32_t& value_size)
//{
//  value_size = 4; // Size in bytes of the nlist n_value member
//  return  GetSectionInfo(GetSectionIndexByName(".stab"), NULL, NULL, NULL, NULL, NULL, NULL, &stabs_data, NULL) &&
//          GetSectionInfo(GetSectionIndexByName(".stabstr"), NULL, NULL, NULL, NULL, NULL, NULL, &stabstr_data, NULL);
//}
//
//===----------------------------------------------------------------------===//
// Dump
//
// Dump the specifics of the runtime file container (such as any headers
// segments, sections, etc).
//----------------------------------------------------------------------
void
ObjectFileELF::Dump(Stream *s)
{
    DumpELFHeader(s, m_header);
    s->EOL();
    DumpELFProgramHeaders(s);
    s->EOL();
    DumpELFSectionHeaders(s);
    s->EOL();
    SectionList *section_list = GetSectionList();
    if (section_list)
        section_list->Dump(s, NULL, true);
    Symtab *symtab = GetSymtab();
    if (symtab)
        symtab->Dump(s, NULL);
    s->EOL();
}

//----------------------------------------------------------------------
// DumpELFHeader
//
// Dump the ELF header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFHeader(Stream *s, const Elf32_Ehdr& header)
{

    s->PutCString ("ELF Header\n");
    s->Printf ("e_ident[EI_MAG0   ] = 0x%2.2x\n", header.e_ident[EI_MAG0]);
    s->Printf ("e_ident[EI_MAG1   ] = 0x%2.2x '%c'\n", header.e_ident[EI_MAG1], header.e_ident[EI_MAG1]);
    s->Printf ("e_ident[EI_MAG2   ] = 0x%2.2x '%c'\n", header.e_ident[EI_MAG2], header.e_ident[EI_MAG2]);
    s->Printf ("e_ident[EI_MAG3   ] = 0x%2.2x '%c'\n", header.e_ident[EI_MAG3], header.e_ident[EI_MAG3]);
    s->Printf ("e_ident[EI_CLASS  ] = 0x%2.2x\n", header.e_ident[EI_CLASS]);
    s->Printf ("e_ident[EI_DATA   ] = 0x%2.2x ", header.e_ident[EI_DATA]);
    DumpELFHeader_e_ident_EI_DATA(s, header.e_ident[EI_DATA]);
    s->Printf ("\ne_ident[EI_VERSION] = 0x%2.2x\n", header.e_ident[EI_VERSION]);
    s->Printf ("e_ident[EI_PAD    ] = 0x%2.2x\n", header.e_ident[EI_PAD]);

    s->Printf("e_type      = 0x%4.4x ", header.e_type);
    DumpELFHeader_e_type(s, header.e_type);
    s->Printf("\ne_machine   = 0x%4.4x\n", header.e_machine);
    s->Printf("e_version   = 0x%8.8x\n", header.e_version);
    s->Printf("e_entry     = 0x%8.8x\n", header.e_entry);
    s->Printf("e_phoff     = 0x%8.8x\n", header.e_phoff);
    s->Printf("e_shoff     = 0x%8.8x\n", header.e_shoff);
    s->Printf("e_flags     = 0x%8.8x\n", header.e_flags);
    s->Printf("e_ehsize    = 0x%4.4x\n", header.e_ehsize);
    s->Printf("e_phentsize = 0x%4.4x\n", header.e_phentsize);
    s->Printf("e_phnum     = 0x%4.4x\n", header.e_phnum);
    s->Printf("e_shentsize = 0x%4.4x\n", header.e_shentsize);
    s->Printf("e_shnum     = 0x%4.4x\n", header.e_shnum);
    s->Printf("e_shstrndx  = 0x%4.4x\n", header.e_shstrndx);
}

//----------------------------------------------------------------------
// DumpELFHeader_e_type
//
// Dump an token value for the ELF header member e_type
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFHeader_e_type(Stream *s, uint16_t e_type)
{
    switch (e_type)
    {
    case ET_NONE:   *s << "ET_NONE"; break;
    case ET_REL:    *s << "ET_REL"; break;
    case ET_EXEC:   *s << "ET_EXEC"; break;
    case ET_DYN:    *s << "ET_DYN"; break;
    case ET_CORE:   *s << "ET_CORE"; break;
    default:
        break;
    }
}

//----------------------------------------------------------------------
// DumpELFHeader_e_ident_EI_DATA
//
// Dump an token value for the ELF header member e_ident[EI_DATA]
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFHeader_e_ident_EI_DATA(Stream *s, uint16_t ei_data)
{
    switch (ei_data)
    {
    case ELFDATANONE:   *s << "ELFDATANONE"; break;
    case ELFDATA2LSB:   *s << "ELFDATA2LSB - Little Endian"; break;
    case ELFDATA2MSB:   *s << "ELFDATA2MSB - Big Endian"; break;
    default:
        break;
    }
}


//----------------------------------------------------------------------
// DumpELFProgramHeader
//
// Dump a single ELF program header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeader(Stream *s, const Elf32_Phdr& ph)
{
    DumpELFProgramHeader_p_type(s, ph.p_type);
    s->Printf(" %8.8x %8.8x %8.8x %8.8x %8.8x %8.8x (", ph.p_offset, ph.p_vaddr, ph.p_paddr, ph.p_filesz, ph.p_memsz, ph.p_flags);
    DumpELFProgramHeader_p_flags(s, ph.p_flags);
    s->Printf(") %8.8x", ph.p_align);
}

//----------------------------------------------------------------------
// DumpELFProgramHeader_p_type
//
// Dump an token value for the ELF program header member p_type which
// describes the type of the program header
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeader_p_type(Stream *s, Elf32_Word p_type)
{
    const int kStrWidth = 10;
    switch (p_type)
    {
    CASE_AND_STREAM(s, PT_NULL      , kStrWidth);
    CASE_AND_STREAM(s, PT_LOAD      , kStrWidth);
    CASE_AND_STREAM(s, PT_DYNAMIC   , kStrWidth);
    CASE_AND_STREAM(s, PT_INTERP    , kStrWidth);
    CASE_AND_STREAM(s, PT_NOTE      , kStrWidth);
    CASE_AND_STREAM(s, PT_SHLIB     , kStrWidth);
    CASE_AND_STREAM(s, PT_PHDR      , kStrWidth);
    default:
        s->Printf("0x%8.8x%*s", p_type, kStrWidth - 10, "");
        break;
    }
}


//----------------------------------------------------------------------
// DumpELFProgramHeader_p_flags
//
// Dump an token value for the ELF program header member p_flags
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeader_p_flags(Stream *s, Elf32_Word p_flags)
{
    *s  << ((p_flags & PF_X) ? "PF_X" : "    ")
        << (((p_flags & PF_X) && (p_flags & PF_W)) ? '+' : ' ')
        << ((p_flags & PF_W) ? "PF_W" : "    ")
        << (((p_flags & PF_W) && (p_flags & PF_R)) ? '+' : ' ')
        << ((p_flags & PF_R) ? "PF_R" : "    ");
}

//----------------------------------------------------------------------
// DumpELFProgramHeaders
//
// Dump all of the ELF program header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeaders(Stream *s)
{
    if (ParseProgramHeaders())
    {
        s->PutCString("Program Headers\n");
        s->PutCString("IDX  p_type     p_offset p_vaddr  p_paddr  p_filesz p_memsz  p_flags                   p_align\n");
        s->PutCString("==== ---------- -------- -------- -------- -------- -------- ------------------------- --------\n");

        uint32_t idx = 0;
        ProgramHeaderCollConstIter pos;

        for (pos = m_program_headers.begin(); pos != m_program_headers.end(); ++pos, ++idx)
        {
            s->Printf ("[%2u] ", idx);
            ObjectFileELF::DumpELFProgramHeader(s, *pos);
            s->EOL();
        }
    }
}


//----------------------------------------------------------------------
// DumpELFSectionHeader
//
// Dump a single ELF section header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeader(Stream *s, const Elf32_Shdr& sh)
{
    s->Printf ("%8.8x ", sh.sh_name);
    DumpELFSectionHeader_sh_type(s, sh.sh_type);
    s->Printf (" %8.8x (", sh.sh_flags);
    DumpELFSectionHeader_sh_flags(s, sh.sh_flags);
    s->Printf (") %8.8x %8.8x %8.8x %8.8x %8.8x %8.8x %8.8x",
                sh.sh_addr, sh.sh_offset, sh.sh_size, sh.sh_link, sh.sh_info, sh.sh_addralign, sh.sh_entsize);
}

//----------------------------------------------------------------------
// DumpELFSectionHeader_sh_type
//
// Dump an token value for the ELF section header member sh_type which
// describes the type of the section
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeader_sh_type(Stream *s, Elf32_Word sh_type)
{
    const int kStrWidth = 12;
    switch (sh_type)
    {
    CASE_AND_STREAM(s, SHT_NULL     , kStrWidth);
    CASE_AND_STREAM(s, SHT_PROGBITS , kStrWidth);
    CASE_AND_STREAM(s, SHT_SYMTAB   , kStrWidth);
    CASE_AND_STREAM(s, SHT_STRTAB   , kStrWidth);
    CASE_AND_STREAM(s, SHT_RELA     , kStrWidth);
    CASE_AND_STREAM(s, SHT_HASH     , kStrWidth);
    CASE_AND_STREAM(s, SHT_DYNAMIC  , kStrWidth);
    CASE_AND_STREAM(s, SHT_NOTE     , kStrWidth);
    CASE_AND_STREAM(s, SHT_NOBITS   , kStrWidth);
    CASE_AND_STREAM(s, SHT_REL      , kStrWidth);
    CASE_AND_STREAM(s, SHT_SHLIB    , kStrWidth);
    CASE_AND_STREAM(s, SHT_DYNSYM   , kStrWidth);
    CASE_AND_STREAM(s, SHT_LOPROC   , kStrWidth);
    CASE_AND_STREAM(s, SHT_HIPROC   , kStrWidth);
    CASE_AND_STREAM(s, SHT_LOUSER   , kStrWidth);
    CASE_AND_STREAM(s, SHT_HIUSER   , kStrWidth);
    default:
        s->Printf("0x%8.8x%*s", sh_type, kStrWidth - 10, "");
        break;
    }
}


//----------------------------------------------------------------------
// DumpELFSectionHeader_sh_flags
//
// Dump an token value for the ELF section header member sh_flags
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeader_sh_flags(Stream *s, Elf32_Word sh_flags)
{
    *s  << ((sh_flags & SHF_WRITE) ? "WRITE" : "     ")
        << (((sh_flags & SHF_WRITE) && (sh_flags & SHF_ALLOC)) ? '+' : ' ')
        << ((sh_flags & SHF_ALLOC) ? "ALLOC" : "     ")
        << (((sh_flags & SHF_ALLOC) && (sh_flags & SHF_EXECINSTR)) ? '+' : ' ')
        << ((sh_flags & SHF_EXECINSTR) ? "EXECINSTR" : "         ");
}
//----------------------------------------------------------------------
// DumpELFSectionHeaders
//
// Dump all of the ELF section header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeaders(Stream *s)
{
    if (ParseSectionHeaders() && GetSectionHeaderStringTable())
    {
        s->PutCString("Section Headers\n");
        s->PutCString("IDX  name     type         flags                            addr     offset   size     link     info     addralgn entsize  Name\n");
        s->PutCString("==== -------- ------------ -------------------------------- -------- -------- -------- -------- -------- -------- -------- ====================\n");

        uint32_t idx = 0;
        SectionHeaderCollConstIter pos;

        for (pos = m_section_headers.begin(); pos != m_section_headers.end(); ++pos, ++idx)
        {
            s->Printf ("[%2u] ", idx);
            ObjectFileELF::DumpELFSectionHeader(s, *pos);
            const char* section_name = m_shstr_data.PeekCStr(pos->sh_name);
            if (section_name)
                *s << ' ' << section_name << "\n";
        }
    }
}

bool
ObjectFileELF::GetTargetTriple (ConstString &target_triple)
{
    static ConstString g_target_triple;

    if (g_target_triple)
    {
        target_triple = g_target_triple;
    }
    else
    {
        std::string triple;
        switch (m_header.e_machine)
        {
        case EM_SPARC:  triple.assign("sparc-"); break;
        case EM_386:    triple.assign("i386-"); break;
        case EM_68K:    triple.assign("68k-"); break;
        case EM_88K:    triple.assign("88k-"); break;
        case EM_860:    triple.assign("i860-"); break;
        case EM_MIPS:   triple.assign("mips-"); break;
        case EM_PPC:    triple.assign("powerpc-"); break;
        case EM_PPC64:  triple.assign("powerpc64-"); break;
        case EM_ARM:    triple.assign("arm-"); break;
        }
        // TODO: determine if there is a vendor in the ELF? Default to "linux" for now
        triple += "linux-";
        // TODO: determine if there is an OS in the ELF? Default to "gnu" for now
        triple += "gnu";
        g_target_triple.SetCString(triple.c_str());
        target_triple = g_target_triple;
    }
    return !target_triple.IsEmpty();
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ObjectFileELF::GetPluginName()
{
    return "ObjectFileELF";
}

const char *
ObjectFileELF::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjectFileELF::GetPluginVersion()
{
    return 1;
}

void
ObjectFileELF::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
ObjectFileELF::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
ObjectFileELF::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}



