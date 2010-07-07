//===-- ObjectFileELF64.cpp ----------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjectFileELF64.h"

#include <cassert>

#include <algorithm>

#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Stream.h"

#define CASE_AND_STREAM(s, def, width)                  \
    case def: s->Printf("%-*s", width, #def); break;

using namespace lldb;
using namespace lldb_private;

void
ObjectFileELF64::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  CreateInstance);
}

void
ObjectFileELF64::Terminate()
{
    PluginManager::UnregisterPlugin(CreateInstance);
}

const char *
ObjectFileELF64::GetPluginNameStatic()
{
    return "object-file.elf64";
}

const char *
ObjectFileELF64::GetPluginDescriptionStatic()
{
    return "ELF object file reader (64-bit).";
}

ObjectFile *
ObjectFileELF64::CreateInstance(Module *module,
                                DataBufferSP &dataSP,
                                const FileSpec *file, addr_t offset,
                                addr_t length)
{
    if (ObjectFileELF64::MagicBytesMatch(dataSP))
    {
        std::auto_ptr<ObjectFile> objfile_ap(
            new ObjectFileELF64(module, dataSP, file, offset, length));
        if (objfile_ap->ParseHeader())
            return objfile_ap.release();
    }
    return NULL;
}

bool
ObjectFileELF64::MagicBytesMatch(DataBufferSP& data_sp)
{
    if (data_sp && data_sp->GetByteSize() > EI_PAD)
    {
        const uint8_t* magic = data_sp->GetBytes();
        if (magic != NULL)
        {
            bool have_magic = (magic[EI_MAG0] == 0x7f &&
                               magic[EI_MAG1] == 'E'  &&
                               magic[EI_MAG2] == 'L'  &&
                               magic[EI_MAG3] == 'F');
        
            bool have_64bit = magic[EI_CLASS] == ELFCLASS64;
        
            return have_magic && have_64bit;
        }
    }
    return false;
}

ObjectFileELF64::ObjectFileELF64(Module* module, DataBufferSP& dataSP,
                                 const FileSpec* file, addr_t offset,
                                 addr_t length)
    : ObjectFile(module, file, offset, length, dataSP),
      m_header(),
      m_program_headers(),
      m_section_headers(),
      m_sections_ap(),
      m_symtab_ap(),
      m_filespec_ap(),
      m_shstr_data()
{
    if (file)
        m_file = *file;
    ::memset(&m_header, 0, sizeof(m_header));
}


ObjectFileELF64::~ObjectFileELF64()
{
}

ByteOrder
ObjectFileELF64::GetByteOrder() const
{
    if (m_header.e_ident[EI_DATA] == ELFDATA2MSB)
        return eByteOrderBig;
    if (m_header.e_ident[EI_DATA] == ELFDATA2LSB)
        return eByteOrderLittle;
    return eByteOrderInvalid;
}

size_t
ObjectFileELF64::GetAddressByteSize() const
{
    return m_data.GetAddressByteSize();
}

unsigned
ObjectFileELF64::SectionIndex(const SectionHeaderCollIter &I)
{
    return std::distance(m_section_headers.begin(), I) + 1;
}

unsigned
ObjectFileELF64::SectionIndex(const SectionHeaderCollConstIter &I) const
{
    return std::distance(m_section_headers.begin(), I) + 1;
}

bool
ObjectFileELF64::ParseHeader()
{
    m_data.SetAddressByteSize(8);
    uint32_t offset = GetOffset();
    if (m_data.GetU8(&offset, m_header.e_ident, EI_NIDENT) == NULL)
        return false;

    m_data.SetByteOrder(GetByteOrder());

    // Read e_type and e_machine.
    if (m_data.GetU16(&offset, &m_header.e_type, 2) == NULL)
        return false;

    // Read e_version.
    if (m_data.GetU32(&offset, &m_header.e_version, 1) == NULL)
        return false;

    // Read e_entry, e_phoff and e_shoff.
    if (m_data.GetU64(&offset, &m_header.e_entry, 3) == NULL)
        return false;

    // Read e_flags.
    if (m_data.GetU32(&offset, &m_header.e_flags, 1) == NULL)
        return false;

    // Read e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum and e_shstrndx.
    if (m_data.GetU16(&offset, &m_header.e_ehsize, 6) == NULL)
        return false;

    return true;
}

bool
ObjectFileELF64::GetUUID(UUID* uuid)
{
    // FIXME: Return MD5 sum here.  See comment in ObjectFile.h.
    return false;
}

uint32_t
ObjectFileELF64::GetDependentModules(FileSpecList &files)
{
    size_t num_modules = ParseDependentModules();
    uint32_t num_specs = 0;

    for (unsigned i = 0; i < num_modules; ++i)
    {
        if (files.AppendIfUnique(m_filespec_ap->GetFileSpecAtIndex(i)))
            num_specs++;
    }

    return num_specs;
}

//----------------------------------------------------------------------
// ParseDependentModules
//----------------------------------------------------------------------
size_t
ObjectFileELF64::ParseDependentModules()
{
    if (m_filespec_ap.get())
        return m_filespec_ap->GetSize();

    m_filespec_ap.reset(new FileSpecList());

    if (!(ParseSectionHeaders() && GetSectionHeaderStringTable()))
        return 0;

    // Locate the dynamic table.
    user_id_t dynsym_id = 0;
    user_id_t dynstr_id = 0;
    for (SectionHeaderCollIter I = m_section_headers.begin();
         I != m_section_headers.end(); ++I)
    {
        if (I->sh_type == SHT_DYNAMIC)
        {
            dynsym_id = SectionIndex(I);
            dynstr_id = I->sh_link + 1; // Section ID's are 1 based.
            break;
        }
    }

    if (!(dynsym_id && dynstr_id))
        return 0;

    SectionList *section_list = GetSectionList();
    if (!section_list)
        return 0;

    // Resolve and load the dynamic table entries and corresponding string
    // table.
    Section *dynsym = section_list->FindSectionByID(dynsym_id).get();
    Section *dynstr = section_list->FindSectionByID(dynstr_id).get();
    if (!(dynsym && dynstr))
        return 0;

    DataExtractor dynsym_data;
    DataExtractor dynstr_data;
    if (dynsym->ReadSectionDataFromObjectFile(this, dynsym_data) &&
        dynstr->ReadSectionDataFromObjectFile(this, dynstr_data))
    {
        Elf64_Dyn symbol;
        const unsigned num_syms = dynsym_data.GetByteSize() / sizeof(Elf64_Dyn);
        unsigned offset = 0;

        // The only type of entries we are concerned with are tagged DT_NEEDED,
        // yielding the name of a required library.
        for (unsigned i = 0; i < num_syms; ++i)
        {
            if (!dynsym_data.ValidOffsetForDataOfSize(offset, sizeof(Elf64_Dyn)))
                break;

            symbol.d_tag = dynsym_data.GetU64(&offset);
            symbol.d_un.d_val = dynsym_data.GetU64(&offset);

            if (symbol.d_tag != DT_NEEDED)
                continue;

            const char *lib_name = dynstr_data.PeekCStr(symbol.d_un.d_val);
            m_filespec_ap->Append(FileSpec(lib_name));
        }
    }

    return m_filespec_ap->GetSize();
}

//----------------------------------------------------------------------
// ParseProgramHeaders
//----------------------------------------------------------------------
size_t
ObjectFileELF64::ParseProgramHeaders()
{
    // We have already parsed the program headers
    if (!m_program_headers.empty())
        return m_program_headers.size();

    // If there are no program headers to read we are done.
    if (m_header.e_phnum == 0)
        return 0;

    m_program_headers.resize(m_header.e_phnum);
    if (m_program_headers.size() != m_header.e_phnum)
        return 0;

    const size_t ph_size = m_header.e_phnum * m_header.e_phentsize;
    const Elf64_Off ph_offset = m_offset + m_header.e_phoff;
    DataBufferSP buffer_sp(m_file.ReadFileContents(ph_offset, ph_size));

    if (buffer_sp.get() == NULL || buffer_sp->GetByteSize() != ph_size)
        return 0;

    DataExtractor data(buffer_sp, m_data.GetByteOrder(),
                       m_data.GetAddressByteSize());

    uint32_t idx;
    uint32_t offset;
    for (idx = 0, offset = 0; idx < m_header.e_phnum; ++idx)
    {
        if (data.GetU32(&offset, &m_program_headers[idx], 8) == NULL)
            return 0;
    }

    if (idx < m_program_headers.size())
        m_program_headers.resize(idx);

    return m_program_headers.size();
}

//----------------------------------------------------------------------
// ParseSectionHeaders
//----------------------------------------------------------------------
size_t
ObjectFileELF64::ParseSectionHeaders()
{
    // We have already parsed the section headers
    if (!m_section_headers.empty())
        return m_section_headers.size();

    // If there are no section headers we are done.
    if (m_header.e_shnum == 0)
        return 0;

    m_section_headers.resize(m_header.e_shnum);
    if (m_section_headers.size() != m_header.e_shnum)
        return 0;

    const size_t sh_size = m_header.e_shnum * m_header.e_shentsize;
    const Elf64_Off sh_offset = m_offset + m_header.e_shoff;
    DataBufferSP buffer_sp(m_file.ReadFileContents(sh_offset, sh_size));

    if (buffer_sp.get() == NULL || buffer_sp->GetByteSize() != sh_size)
        return 0;

    DataExtractor data(buffer_sp,
                       m_data.GetByteOrder(),
                       m_data.GetAddressByteSize());

    uint32_t idx;
    uint32_t offset;
    for (idx = 0, offset = 0; idx < m_header.e_shnum; ++idx)
    {
        // Read sh_name and sh_type.
        if (data.GetU32(&offset, &m_section_headers[idx].sh_name, 2) == NULL)
            break;

        // Read sh_flags, sh_addr, sh_offset and sh_size.
        if (data.GetU64(&offset, &m_section_headers[idx].sh_flags, 4) == NULL)
            break;

        // Read sh_link and sh_info.
        if (data.GetU32(&offset, &m_section_headers[idx].sh_link, 2) == NULL)
            break;

        // Read sh_addralign and sh_entsize.
        if (data.GetU64(&offset, &m_section_headers[idx].sh_addralign, 2) == NULL)
            break;
    }
    if (idx < m_section_headers.size())
        m_section_headers.resize(idx);

    return m_section_headers.size();
}

size_t
ObjectFileELF64::GetSectionHeaderStringTable()
{
    if (m_shstr_data.GetByteSize() == 0)
    {
        if (m_header.e_shstrndx && m_header.e_shstrndx < m_section_headers.size())
        {
            const Elf64_Shdr &sheader = m_section_headers[m_header.e_shstrndx];
            const size_t byte_size = sheader.sh_size;
            const Elf64_Off offset = m_offset + sheader.sh_offset;
            DataBufferSP buffer_sp(m_file.ReadFileContents(offset, byte_size));

            if (buffer_sp.get() == NULL || buffer_sp->GetByteSize() != byte_size)
                return 0;

            m_shstr_data.SetData(buffer_sp);
        }
    }
    return m_shstr_data.GetByteSize();
}

uint32_t
ObjectFileELF64::GetSectionIndexByName(const char *name)
{
    if (!(ParseSectionHeaders() && GetSectionHeaderStringTable()))
        return UINT32_MAX;

    // Search the collection of section headers for one with a matching name.
    for (SectionHeaderCollIter I = m_section_headers.begin();
         I != m_section_headers.end(); ++I)
    {
        const char *sectionName = m_shstr_data.PeekCStr(I->sh_name);

        if (!sectionName)
            return UINT32_MAX;

        if (strcmp(name, sectionName) != 0)
            continue;

        return SectionIndex(I);
    }

    return UINT32_MAX;
}

SectionList *
ObjectFileELF64::GetSectionList()
{
    if (m_sections_ap.get())
        return m_sections_ap.get();

    if (ParseSectionHeaders() && GetSectionHeaderStringTable())
    {
        m_sections_ap.reset(new SectionList());

        for (SectionHeaderCollIter I = m_section_headers.begin();
             I != m_section_headers.end(); ++I)
        {
            const Elf64_Shdr &header = *I;

            ConstString name(m_shstr_data.PeekCStr(header.sh_name));
            uint64_t size = header.sh_type == SHT_NOBITS ? 0 : header.sh_size;

            SectionSP section(new Section(
                0,                 // Parent section.
                GetModule(),       // Module to which this section belongs.
                SectionIndex(I),   // Section ID.
                name,              // Section name.
                eSectionTypeOther, // FIXME: Fill in as appropriate.
                header.sh_addr,    // VM address.
                header.sh_size,    // VM size in bytes of this section.
                header.sh_offset,  // Offset of this section in the file.
                size,              // Size of the section as found in the file.
                header.sh_flags)); // Flags for this section.

            m_sections_ap->AddSection(section);
        }
    }

    return m_sections_ap.get();
}

static void
ParseSymbols(Symtab *symtab, SectionList *section_list,
             const Elf64_Shdr &symtab_shdr,
             const DataExtractor &symtab_data,
             const DataExtractor &strtab_data)
{
    assert (sizeof(Elf64_Sym) == symtab_shdr.sh_entsize);
    const unsigned numSymbols = symtab_data.GetByteSize() / sizeof(Elf64_Sym);
    unsigned offset = 0;
    Elf64_Sym symbol;

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

    for (unsigned i = 0; i < numSymbols; ++i)
    {
        if (!symtab_data.ValidOffsetForDataOfSize(offset, sizeof(Elf64_Sym)))
            break;

        symbol.st_name  = symtab_data.GetU32(&offset);
        symbol.st_info  = symtab_data.GetU8(&offset);
        symbol.st_other = symtab_data.GetU8(&offset);
        symbol.st_shndx = symtab_data.GetU16(&offset);
        symbol.st_value = symtab_data.GetU64(&offset);
        symbol.st_size  = symtab_data.GetU64(&offset);

        Section *symbol_section = NULL;
        SymbolType symbol_type = eSymbolTypeInvalid;
        Elf64_Half symbol_idx = symbol.st_shndx;

        switch (symbol_idx)
        {
        case SHN_ABS:
            symbol_type = eSymbolTypeAbsolute;
            break;
        case SHN_UNDEF:
            symbol_type = eSymbolTypeUndefined;
            break;
        default:
            symbol_section = section_list->GetSectionAtIndex(symbol_idx).get();
            break;
        }

        switch (ELF_ST_TYPE(symbol.st_info))
        {
        default:
        case STT_NOTYPE:
            // The symbol's type is not specified.
            break;

        case STT_OBJECT:
            // The symbol is associated with a data object, such as a variable,
            // an array, etc.
            symbol_type = eSymbolTypeData;
            break;

        case STT_FUNC:
            // The symbol is associated with a function or other executable code.
            symbol_type = eSymbolTypeCode;
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
            symbol_type = eSymbolTypeObjectFile;
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
                else if (sect_name == data_section_name ||
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
        bool is_global = ELF_ST_BIND(symbol.st_info) == STB_GLOBAL;
        uint32_t flags = symbol.st_other << 8 | symbol.st_info;

        Symbol dc_symbol(
            i,               // ID is the original symbol table index.
            symbol_name,     // symbol name.
            false,           // Is the symbol name mangled?
            symbol_type,     // type of this symbol
            is_global,       // Is this globally visible?
            false,           // Is this symbol debug info?
            false,           // Is this symbol a trampoline?
            false,           // Is this symbol artificial?
            symbol_section,  // Section in which this symbol is defined or null.
            symbol_value,    // Offset in section or symbol value.
            symbol.st_size,  // size in bytes of this symbol.
            flags);          // Symbol flags.
        symtab->AddSymbol(dc_symbol);
    }
}

void
ObjectFileELF64::ParseSymbolTable(Symtab *symbol_table,
                                  const Elf64_Shdr &symtab_hdr,
                                  user_id_t symtab_id)
{
    assert(symtab_hdr.sh_type == SHT_SYMTAB || 
           symtab_hdr.sh_type == SHT_DYNSYM);

    // Parse in the section list if needed.
    SectionList *section_list = GetSectionList();
    if (!section_list)
        return;

    // Section ID's are ones based.
    user_id_t strtab_id = symtab_hdr.sh_link + 1;

    Section *symtab = section_list->FindSectionByID(symtab_id).get();
    Section *strtab = section_list->FindSectionByID(strtab_id).get();
    if (symtab && strtab)
    {
        DataExtractor symtab_data;
        DataExtractor strtab_data;
        if (symtab->ReadSectionDataFromObjectFile(this, symtab_data) &&
            strtab->ReadSectionDataFromObjectFile(this, strtab_data))
        {
            ParseSymbols(symbol_table, section_list, symtab_hdr,
                         symtab_data, strtab_data);
        }
    }
}

Symtab *
ObjectFileELF64::GetSymtab()
{
    if (m_symtab_ap.get())
        return m_symtab_ap.get();

    Symtab *symbol_table = new Symtab(this);
    m_symtab_ap.reset(symbol_table);

    if (!(ParseSectionHeaders() && GetSectionHeaderStringTable()))
        return symbol_table;

    // Locate and parse all linker symbol tables.
    for (SectionHeaderCollIter I = m_section_headers.begin();
         I != m_section_headers.end(); ++I)
    {
        if (I->sh_type == SHT_SYMTAB || I->sh_type == SHT_DYNSYM)
        {
            const Elf64_Shdr &symtab_section = *I;
            user_id_t section_id = SectionIndex(I);
            ParseSymbolTable(symbol_table, symtab_section, section_id);
        }
    }

    return symbol_table;
}

//===----------------------------------------------------------------------===//
// Dump
//
// Dump the specifics of the runtime file container (such as any headers
// segments, sections, etc).
// ----------------------------------------------------------------------
void
ObjectFileELF64::Dump(Stream *s)
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
    DumpDependentModules(s);
    s->EOL();
}

//----------------------------------------------------------------------
// DumpELFHeader
//
// Dump the ELF header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF64::DumpELFHeader(Stream *s, const Elf64_Ehdr& header)
{

    s->PutCString("ELF Header\n");
    s->Printf("e_ident[EI_MAG0   ] = 0x%2.2x\n", header.e_ident[EI_MAG0]);
    s->Printf("e_ident[EI_MAG1   ] = 0x%2.2x '%c'\n",
              header.e_ident[EI_MAG1], header.e_ident[EI_MAG1]);
    s->Printf("e_ident[EI_MAG2   ] = 0x%2.2x '%c'\n",
              header.e_ident[EI_MAG2], header.e_ident[EI_MAG2]);
    s->Printf("e_ident[EI_MAG3   ] = 0x%2.2x '%c'\n",
              header.e_ident[EI_MAG3], header.e_ident[EI_MAG3]);

    s->Printf("e_ident[EI_CLASS  ] = 0x%2.2x\n", header.e_ident[EI_CLASS]);
    s->Printf("e_ident[EI_DATA   ] = 0x%2.2x ", header.e_ident[EI_DATA]);
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
ObjectFileELF64::DumpELFHeader_e_type(Stream *s, Elf64_Half e_type)
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
ObjectFileELF64::DumpELFHeader_e_ident_EI_DATA(Stream *s, unsigned char ei_data)
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
ObjectFileELF64::DumpELFProgramHeader(Stream *s, const Elf64_Phdr &ph)
{
    DumpELFProgramHeader_p_type(s, ph.p_type);
    s->Printf(" %8.8x %8.8x %8.8x %8.8x %8.8x %8.8x (",
              ph.p_offset, ph.p_vaddr, ph.p_paddr, ph.p_filesz, ph.p_memsz,
              ph.p_flags);
    DumpELFProgramHeader_p_flags(s, ph.p_flags);
    s->Printf(") %8.8x", ph.p_align);
}

//----------------------------------------------------------------------
// DumpELFProgramHeader_p_type
//
// Dump an token value for the ELF program header member p_type which
// describes the type of the program header
// ----------------------------------------------------------------------
void
ObjectFileELF64::DumpELFProgramHeader_p_type(Stream *s, Elf64_Word p_type)
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
ObjectFileELF64::DumpELFProgramHeader_p_flags(Stream *s, Elf64_Word p_flags)
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
ObjectFileELF64::DumpELFProgramHeaders(Stream *s)
{
    if (ParseProgramHeaders())
    {
        s->PutCString("Program Headers\n");
        s->PutCString("IDX  p_type     p_offset p_vaddr  p_paddr  "
                      "p_filesz p_memsz  p_flags                   p_align\n");
        s->PutCString("==== ---------- -------- -------- -------- "
                      "-------- -------- ------------------------- --------\n");

        uint32_t idx = 0;
        for (ProgramHeaderCollConstIter I = m_program_headers.begin();
             I != m_program_headers.end(); ++I, ++idx)
        {
            s->Printf("[%2u] ", idx);
            ObjectFileELF64::DumpELFProgramHeader(s, *I);
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
ObjectFileELF64::DumpELFSectionHeader(Stream *s, const Elf64_Shdr &sh)
{
    s->Printf("%8.8x ", sh.sh_name);
    DumpELFSectionHeader_sh_type(s, sh.sh_type);
    s->Printf(" %8.8x (", sh.sh_flags);
    DumpELFSectionHeader_sh_flags(s, sh.sh_flags);
    s->Printf(") %8.8x %8.8x %8.8x %8.8x %8.8x %8.8x %8.8x",
              sh.sh_addr, sh.sh_offset, sh.sh_size, sh.sh_link, sh.sh_info,
              sh.sh_addralign, sh.sh_entsize);
}

//----------------------------------------------------------------------
// DumpELFSectionHeader_sh_type
//
// Dump an token value for the ELF section header member sh_type which
// describes the type of the section
//----------------------------------------------------------------------
void
ObjectFileELF64::DumpELFSectionHeader_sh_type(Stream *s, Elf64_Word sh_type)
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
ObjectFileELF64::DumpELFSectionHeader_sh_flags(Stream *s, Elf64_Word sh_flags)
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
ObjectFileELF64::DumpELFSectionHeaders(Stream *s)
{
    if (!(ParseSectionHeaders() && GetSectionHeaderStringTable()))
        return;

    s->PutCString("Section Headers\n");
    s->PutCString("IDX  name     type         flags                            "
                  "addr     offset   size     link     info     addralgn "
                  "entsize  Name\n");
    s->PutCString("==== -------- ------------ -------------------------------- "
                  "-------- -------- -------- -------- -------- -------- "
                  "-------- ====================\n");

    uint32_t idx = 0;
    for (SectionHeaderCollConstIter I = m_section_headers.begin();
         I != m_section_headers.end(); ++I, ++idx)
    {
        s->Printf("[%2u] ", idx);
        ObjectFileELF64::DumpELFSectionHeader(s, *I);
        const char* section_name = m_shstr_data.PeekCStr(I->sh_name);
        if (section_name)
            *s << ' ' << section_name << "\n";
    }
}

void
ObjectFileELF64::DumpDependentModules(lldb_private::Stream *s)
{
    size_t num_modules = ParseDependentModules();

    if (num_modules > 0)
    {
        s->PutCString("Dependent Modules:\n");
        for (unsigned i = 0; i < num_modules; ++i)
        {
            const FileSpec &spec = m_filespec_ap->GetFileSpecAtIndex(i);
            s->Printf("   %s\n", spec.GetFilename().GetCString());
        }
    }
}

bool
ObjectFileELF64::GetTargetTriple(ConstString &target_triple)
{
    static ConstString g_target_triple;

    if (g_target_triple)
    {
        target_triple = g_target_triple;
        return true;
    }

    std::string triple;
    switch (m_header.e_machine)
    {
    default:
        assert(false && "Unexpected machine type.");
        break;
    case EM_SPARC:  triple.assign("sparc-"); break;
    case EM_386:    triple.assign("i386-"); break;
    case EM_68K:    triple.assign("68k-"); break;
    case EM_88K:    triple.assign("88k-"); break;
    case EM_860:    triple.assign("i860-"); break;
    case EM_MIPS:   triple.assign("mips-"); break;
    case EM_PPC:    triple.assign("powerpc-"); break;
    case EM_PPC64:  triple.assign("powerpc64-"); break;
    case EM_ARM:    triple.assign("arm-"); break;
    case EM_X86_64: triple.assign("x86_64-"); break;
    }
    // TODO: determine if there is a vendor in the ELF? Default to "linux" for now
    triple += "linux-";
    // TODO: determine if there is an OS in the ELF? Default to "gnu" for now
    triple += "gnu";
    g_target_triple.SetCString(triple.c_str());
    target_triple = g_target_triple;

    return true;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ObjectFileELF64::GetPluginName()
{
    return "ObjectFileELF64";
}

const char *
ObjectFileELF64::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjectFileELF64::GetPluginVersion()
{
    return 1;
}

void
ObjectFileELF64::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
ObjectFileELF64::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in commands are currently supported.");
    return error;
}

Log *
ObjectFileELF64::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}
