//===-- ObjectFilePECOFF.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjectFilePECOFF.h"

#include "llvm/Support/MachO.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/UUID.h"
#include "lldb/Symbol/ObjectFile.h"

static uint32_t COFFMachineToMachCPU(uint16_t machine);

#define IMAGE_FILE_MACHINE_UNKNOWN      0x0000
#define IMAGE_FILE_MACHINE_AM33         0x01d3  // Matsushita AM33
#define IMAGE_FILE_MACHINE_AMD64        0x8664  // x64
#define IMAGE_FILE_MACHINE_ARM          0x01c0  // ARM little endian
#define IMAGE_FILE_MACHINE_EBC          0x0ebc  // EFI byte code
#define IMAGE_FILE_MACHINE_I386         0x014c  // Intel 386 or later processors and compatible processors
#define IMAGE_FILE_MACHINE_IA64         0x0200  // Intel Itanium processor family
#define IMAGE_FILE_MACHINE_M32R         0x9041  // Mitsubishi M32R little endian
#define IMAGE_FILE_MACHINE_MIPS16       0x0266  // MIPS16
#define IMAGE_FILE_MACHINE_MIPSFPU      0x0366  // MIPS with FPU
#define IMAGE_FILE_MACHINE_MIPSFPU16    0x0466  // MIPS16 with FPU
#define IMAGE_FILE_MACHINE_POWERPC      0x01f0  // Power PC little endian
#define IMAGE_FILE_MACHINE_POWERPCFP    0x01f1  // Power PC with floating point support
#define IMAGE_FILE_MACHINE_R4000        0x0166  // MIPS little endian
#define IMAGE_FILE_MACHINE_SH3          0x01a2  // Hitachi SH3
#define IMAGE_FILE_MACHINE_SH3DSP       0x01a3  // Hitachi SH3 DSP
#define IMAGE_FILE_MACHINE_SH4          0x01a6  // Hitachi SH4
#define IMAGE_FILE_MACHINE_SH5          0x01a8  // Hitachi SH5
#define IMAGE_FILE_MACHINE_THUMB        0x01c2  // Thumb
#define IMAGE_FILE_MACHINE_WCEMIPSV2    0x0169  // MIPS little-endian WCE v2


#define IMAGE_DOS_SIGNATURE             0x5A4D      // MZ
#define IMAGE_OS2_SIGNATURE             0x454E      // NE
#define IMAGE_OS2_SIGNATURE_LE          0x454C      // LE
#define IMAGE_NT_SIGNATURE              0x00004550  // PE00
#define OPT_HEADER_MAGIC_PE32           0x010b
#define OPT_HEADER_MAGIC_PE32_PLUS      0x020b

#define IMAGE_FILE_RELOCS_STRIPPED          0x0001
#define IMAGE_FILE_EXECUTABLE_IMAGE         0x0002
#define IMAGE_FILE_LINE_NUMS_STRIPPED       0x0004
#define IMAGE_FILE_LOCAL_SYMS_STRIPPED      0x0008    
#define IMAGE_FILE_AGGRESSIVE_WS_TRIM       0x0010
#define IMAGE_FILE_LARGE_ADDRESS_AWARE      0x0020
//#define                                   0x0040  // Reserved
#define IMAGE_FILE_BYTES_REVERSED_LO        0x0080
#define IMAGE_FILE_32BIT_MACHINE            0x0100
#define IMAGE_FILE_DEBUG_STRIPPED           0x0200
#define IMAGE_FILE_REMOVABLE_RUN_FROM_SWAP  0x0400
#define IMAGE_FILE_NET_RUN_FROM_SWAP        0x0800
#define IMAGE_FILE_SYSTEM                   0x1000
#define IMAGE_FILE_DLL                      0x2000
#define IMAGE_FILE_UP_SYSTEM_ONLY           0x4000
#define IMAGE_FILE_BYTES_REVERSED_HI        0x8000

using namespace lldb;
using namespace lldb_private;

void
ObjectFilePECOFF::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
ObjectFilePECOFF::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
ObjectFilePECOFF::GetPluginNameStatic()
{
    return "object-file.pe-coff";
}

const char *
ObjectFilePECOFF::GetPluginDescriptionStatic()
{
    return "Portable Executable and Common Object File Format object file reader (32 and 64 bit)";
}


ObjectFile *
ObjectFilePECOFF::CreateInstance (Module* module, DataBufferSP& dataSP, const FileSpec* file, addr_t offset, addr_t length)
{
    if (ObjectFilePECOFF::MagicBytesMatch(dataSP))
    {
        std::auto_ptr<ObjectFile> objfile_ap(new ObjectFilePECOFF (module, dataSP, file, offset, length));
        if (objfile_ap.get() && objfile_ap->ParseHeader())
            return objfile_ap.release();
    }
    return NULL;
}

bool
ObjectFilePECOFF::MagicBytesMatch (DataBufferSP& dataSP)
{
    DataExtractor data(dataSP, eByteOrderLittle, 4);
    uint32_t offset = 0;
    uint16_t magic = data.GetU16 (&offset);
    return magic == IMAGE_DOS_SIGNATURE;
}


ObjectFilePECOFF::ObjectFilePECOFF (Module* module, 
                                    DataBufferSP& dataSP, 
                                    const FileSpec* file, 
                                    addr_t offset, 
                                    addr_t length) :
    ObjectFile (module, file, offset, length, dataSP),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_dos_header (),
    m_coff_header (),
    m_coff_header_opt (),
    m_sect_headers ()
{
    ::memset (&m_dos_header, 0, sizeof(m_dos_header));
    ::memset (&m_coff_header, 0, sizeof(m_coff_header));
    ::memset (&m_coff_header_opt, 0, sizeof(m_coff_header_opt));
}


ObjectFilePECOFF::~ObjectFilePECOFF()
{
}


bool
ObjectFilePECOFF::ParseHeader ()
{
    Mutex::Locker locker(m_mutex);
    m_sect_headers.clear();
    m_data.SetByteOrder (eByteOrderLittle);
    uint32_t offset = 0;
    
    if (ParseDOSHeader())
    {
        offset = m_dos_header.e_lfanew;
        uint32_t pe_signature = m_data.GetU32 (&offset);
        if (pe_signature != IMAGE_NT_SIGNATURE)
            return false;
    }
    if (ParseCOFFHeader(&offset))
    {
        if (m_coff_header.hdrsize > 0)
            ParseCOFFOptionalHeader(&offset);
        ParseSectionHeaders (offset);
        return true;
    }
    return false;
}


ByteOrder
ObjectFilePECOFF::GetByteOrder () const
{
    return eByteOrderLittle;
}

bool
ObjectFilePECOFF::IsExecutable() const
{
    return (m_coff_header.flags & IMAGE_FILE_DLL) == 0;
}

size_t
ObjectFilePECOFF::GetAddressByteSize () const
{
    if (m_coff_header_opt.magic == OPT_HEADER_MAGIC_PE32_PLUS)
        return 8;
    else if (m_coff_header_opt.magic == OPT_HEADER_MAGIC_PE32)
        return 4;
    return 4;
}

//----------------------------------------------------------------------
// NeedsEndianSwap
//
// Return true if an endian swap needs to occur when extracting data 
// from this file.
//----------------------------------------------------------------------
bool 
ObjectFilePECOFF::NeedsEndianSwap() const
{
#if defined(__LITTLE_ENDIAN__)
    return false;
#else
    return true;
#endif
}
//----------------------------------------------------------------------
// ParseDOSHeader
//----------------------------------------------------------------------
bool
ObjectFilePECOFF::ParseDOSHeader ()
{
    bool success = false;
    uint32_t offset = 0;
    success = m_data.ValidOffsetForDataOfSize(0, sizeof(m_dos_header));
    
    if (success)
    {
        m_dos_header.e_magic = m_data.GetU16(&offset); // Magic number
        success = m_dos_header.e_magic == IMAGE_DOS_SIGNATURE;
        
        if (success)
        {
            m_dos_header.e_cblp     = m_data.GetU16(&offset); // Bytes on last page of file
            m_dos_header.e_cp       = m_data.GetU16(&offset); // Pages in file
            m_dos_header.e_crlc     = m_data.GetU16(&offset); // Relocations
            m_dos_header.e_cparhdr  = m_data.GetU16(&offset); // Size of header in paragraphs
            m_dos_header.e_minalloc = m_data.GetU16(&offset); // Minimum extra paragraphs needed
            m_dos_header.e_maxalloc = m_data.GetU16(&offset); // Maximum extra paragraphs needed
            m_dos_header.e_ss       = m_data.GetU16(&offset); // Initial (relative) SS value
            m_dos_header.e_sp       = m_data.GetU16(&offset); // Initial SP value
            m_dos_header.e_csum     = m_data.GetU16(&offset); // Checksum
            m_dos_header.e_ip       = m_data.GetU16(&offset); // Initial IP value
            m_dos_header.e_cs       = m_data.GetU16(&offset); // Initial (relative) CS value
            m_dos_header.e_lfarlc   = m_data.GetU16(&offset); // File address of relocation table
            m_dos_header.e_ovno     = m_data.GetU16(&offset); // Overlay number
            
            m_dos_header.e_res[0]   = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res[1]   = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res[2]   = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res[3]   = m_data.GetU16(&offset); // Reserved words
            
            m_dos_header.e_oemid    = m_data.GetU16(&offset); // OEM identifier (for e_oeminfo)
            m_dos_header.e_oeminfo  = m_data.GetU16(&offset); // OEM information; e_oemid specific
            m_dos_header.e_res2[0]  = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res2[1]  = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res2[2]  = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res2[3]  = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res2[4]  = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res2[5]  = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res2[6]  = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res2[7]  = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res2[8]  = m_data.GetU16(&offset); // Reserved words
            m_dos_header.e_res2[9]  = m_data.GetU16(&offset); // Reserved words
            
            m_dos_header.e_lfanew   = m_data.GetU32(&offset); // File address of new exe header
        }
    }
    if (!success)
        memset(&m_dos_header, 0, sizeof(m_dos_header));
    return success;
}


//----------------------------------------------------------------------
// ParserCOFFHeader
//----------------------------------------------------------------------
bool
ObjectFilePECOFF::ParseCOFFHeader(uint32_t* offset_ptr)
{
    bool success = m_data.ValidOffsetForDataOfSize (*offset_ptr, sizeof(m_coff_header));
    if (success)
    {
        m_coff_header.machine   = m_data.GetU16(offset_ptr);
        m_coff_header.nsects    = m_data.GetU16(offset_ptr);
        m_coff_header.modtime   = m_data.GetU32(offset_ptr);
        m_coff_header.symoff    = m_data.GetU32(offset_ptr);
        m_coff_header.nsyms     = m_data.GetU32(offset_ptr);
        m_coff_header.hdrsize   = m_data.GetU16(offset_ptr);
        m_coff_header.flags     = m_data.GetU16(offset_ptr);
    }
    if (!success)
        memset(&m_coff_header, 0, sizeof(m_coff_header));
    return success;
}

bool
ObjectFilePECOFF::ParseCOFFOptionalHeader(uint32_t* offset_ptr)
{
    bool success = false;
    const uint32_t end_offset = *offset_ptr + m_coff_header.hdrsize;
    if (*offset_ptr < end_offset)
    {
        success = true;
        m_coff_header_opt.magic                         = m_data.GetU16(offset_ptr); 
        m_coff_header_opt.major_linker_version          = m_data.GetU8 (offset_ptr);
        m_coff_header_opt.minor_linker_version          = m_data.GetU8 (offset_ptr);     
        m_coff_header_opt.code_size                     = m_data.GetU32(offset_ptr); 
        m_coff_header_opt.data_size                     = m_data.GetU32(offset_ptr); 
        m_coff_header_opt.bss_size                      = m_data.GetU32(offset_ptr); 
        m_coff_header_opt.entry                         = m_data.GetU32(offset_ptr); 
        m_coff_header_opt.code_offset                   = m_data.GetU32(offset_ptr); 

        const uint32_t addr_byte_size = GetAddressByteSize ();

        if (*offset_ptr < end_offset)
        {
            if (m_coff_header_opt.magic == OPT_HEADER_MAGIC_PE32)
            {
                // PE32 only
                m_coff_header_opt.data_offset               = m_data.GetU32(offset_ptr);                             
            }
            else
                m_coff_header_opt.data_offset = 0;
        
            if (*offset_ptr < end_offset)
            {
                m_coff_header_opt.image_base                    = m_data.GetMaxU64 (offset_ptr, addr_byte_size); 
                m_coff_header_opt.sect_alignment                = m_data.GetU32(offset_ptr); 
                m_coff_header_opt.file_alignment                = m_data.GetU32(offset_ptr); 
                m_coff_header_opt.major_os_system_version       = m_data.GetU16(offset_ptr); 
                m_coff_header_opt.minor_os_system_version       = m_data.GetU16(offset_ptr); 
                m_coff_header_opt.major_image_version           = m_data.GetU16(offset_ptr); 
                m_coff_header_opt.minor_image_version           = m_data.GetU16(offset_ptr); 
                m_coff_header_opt.major_subsystem_version       = m_data.GetU16(offset_ptr); 
                m_coff_header_opt.minor_subsystem_version       = m_data.GetU16(offset_ptr); 
                m_coff_header_opt.reserved1                     = m_data.GetU32(offset_ptr); 
                m_coff_header_opt.image_size                    = m_data.GetU32(offset_ptr); 
                m_coff_header_opt.header_size                   = m_data.GetU32(offset_ptr); 
                m_coff_header_opt.checksym                      = m_data.GetU32(offset_ptr); 
                m_coff_header_opt.subsystem                     = m_data.GetU16(offset_ptr); 
                m_coff_header_opt.dll_flags                     = m_data.GetU16(offset_ptr); 
                m_coff_header_opt.stack_reserve_size            = m_data.GetMaxU64 (offset_ptr, addr_byte_size);
                m_coff_header_opt.stack_commit_size             = m_data.GetMaxU64 (offset_ptr, addr_byte_size);
                m_coff_header_opt.heap_reserve_size             = m_data.GetMaxU64 (offset_ptr, addr_byte_size);
                m_coff_header_opt.heap_commit_size              = m_data.GetMaxU64 (offset_ptr, addr_byte_size);
                m_coff_header_opt.loader_flags                  = m_data.GetU32(offset_ptr); 
                uint32_t num_data_dir_entries = m_data.GetU32(offset_ptr);
                m_coff_header_opt.data_dirs.clear();
                m_coff_header_opt.data_dirs.resize(num_data_dir_entries);
                uint32_t i;
                for (i=0; i<num_data_dir_entries; i++)
                {
                    m_coff_header_opt.data_dirs[i].vmaddr = m_data.GetU32(offset_ptr);
                    m_coff_header_opt.data_dirs[i].vmsize = m_data.GetU32(offset_ptr);
                }
            }
        }
    }
    // Make sure we are on track for section data which follows
    *offset_ptr = end_offset;
    return success;
}


//----------------------------------------------------------------------
// ParseSectionHeaders
//----------------------------------------------------------------------
bool
ObjectFilePECOFF::ParseSectionHeaders (uint32_t section_header_data_offset)
{
    const uint32_t nsects = m_coff_header.nsects;
    m_sect_headers.clear();
    
    if (nsects > 0)
    {
        const uint32_t addr_byte_size = GetAddressByteSize ();
        const size_t section_header_byte_size = nsects * sizeof(section_header_t);
        DataBufferSP section_header_data_sp(m_file.ReadFileContents (section_header_data_offset, section_header_byte_size));
        DataExtractor section_header_data (section_header_data_sp, GetByteOrder(), addr_byte_size);

        uint32_t offset = 0;
        if (section_header_data.ValidOffsetForDataOfSize (offset, section_header_byte_size))
        {
            m_sect_headers.resize(nsects);
            
            for (uint32_t idx = 0; idx<nsects; ++idx)
            {
                const void *name_data = section_header_data.GetData(&offset, 8);
                if (name_data)
                {
                    memcpy(m_sect_headers[idx].name, name_data, 8);
                    m_sect_headers[idx].vmsize  = section_header_data.GetU32(&offset);
                    m_sect_headers[idx].vmaddr  = section_header_data.GetU32(&offset);
                    m_sect_headers[idx].size    = section_header_data.GetU32(&offset);
                    m_sect_headers[idx].offset  = section_header_data.GetU32(&offset);
                    m_sect_headers[idx].reloff  = section_header_data.GetU32(&offset);
                    m_sect_headers[idx].lineoff = section_header_data.GetU32(&offset);
                    m_sect_headers[idx].nreloc  = section_header_data.GetU16(&offset);
                    m_sect_headers[idx].nline   = section_header_data.GetU16(&offset);
                    m_sect_headers[idx].flags   = section_header_data.GetU32(&offset);
                }
            }
        }
    }
    
    return m_sect_headers.empty() == false;
}

bool
ObjectFilePECOFF::GetSectionName(std::string& sect_name, const section_header_t& sect)
{
    if (sect.name[0] == '/')
    {
        uint32_t stroff = strtoul(&sect.name[1], NULL, 10);
        uint32_t string_file_offset = m_coff_header.symoff + (m_coff_header.nsyms * 18) + stroff;
        const char *name = m_data.GetCStr (&string_file_offset);
        if (name)
        {
            sect_name = name;
            return true;
        }
        
        return false;
    }
    sect_name = sect.name;
    return true;
}       

//----------------------------------------------------------------------
// GetNListSymtab
//----------------------------------------------------------------------
Symtab *
ObjectFilePECOFF::GetSymtab()
{
    Mutex::Locker symfile_locker(m_mutex);
    if (m_symtab_ap.get() == NULL)
    {
        SectionList *sect_list = GetSectionList();
        m_symtab_ap.reset(new Symtab(this));
        Mutex::Locker symtab_locker (m_symtab_ap->GetMutex());
        DataExtractor stab_data;
        DataExtractor stabstr_data;
        static ConstString g_stab_sect_name(".stab");
        static ConstString g_stabstr_sect_name(".stabstr");
        const Section *stab_section = sect_list->FindSectionByName (g_stab_sect_name).get();
        if (stab_section == NULL)
            return m_symtab_ap.get();
        const Section *stabstr_section = sect_list->FindSectionByName (g_stabstr_sect_name).get();
        if (stabstr_section == NULL)
            return m_symtab_ap.get();
        uint32_t offset = 0;
        if (stab_section->ReadSectionDataFromObjectFile (this, stab_data) &&
            stabstr_section->ReadSectionDataFromObjectFile (this, stabstr_data))
        {
            const uint32_t num_syms = stab_data.GetByteSize() / sizeof (coff_symbol_t);
            Symbol *symbols = m_symtab_ap->Resize (num_syms);
            for (uint32_t i=0; i<num_syms; ++i)
            {
                coff_symbol_t symbol;
                const void *name_data = stab_data.GetData(&offset, sizeof(symbol.name));
                
                if (name_data == NULL)
                    break;
                memcpy (symbol.name, name_data, sizeof(symbol.name));
                symbol.value = stab_data.GetU32 (&offset);
                symbol.sect = stab_data.GetU16 (&offset);
                symbol.type = stab_data.GetU16 (&offset);
                symbol.storage = stab_data.GetU8 (&offset);
                symbol.naux = stab_data.GetU8 (&offset);		

                symbol.name[sizeof(symbol.name)-1] = '\0';
                Address symbol_addr(sect_list->GetSectionAtIndex(symbol.sect-1).get(), symbol.value);
                symbols[i].GetMangled ().SetValue(symbol.name, symbol.name[0]=='_' && symbol.name[1] == 'Z');
                symbols[i].SetValue(symbol_addr);

                if (symbol.naux > 0)
                    i += symbol.naux;
            }
            
        }
    }
    return m_symtab_ap.get();

}

SectionList *
ObjectFilePECOFF::GetSectionList()
{
    Mutex::Locker symfile_locker(m_mutex);
    if (m_sections_ap.get() == NULL)
    {
        m_sections_ap.reset(new SectionList());
        const uint32_t nsects = m_sect_headers.size();
        Module *module = GetModule();
        for (uint32_t idx = 0; idx<nsects; ++idx)
        {
            std::string sect_name;
            GetSectionName (sect_name, m_sect_headers[idx]);
            ConstString const_sect_name (sect_name.c_str());
            
            // Use a segment ID of the segment index shifted left by 8 so they
            // never conflict with any of the sections.
            SectionSP section_sp (new Section (NULL,
                                               module,                       // Module to which this section belongs
                                               idx + 1,                      // Section ID is the 1 based segment index shifted right by 8 bits as not to collide with any of the 256 section IDs that are possible
                                               const_sect_name,              // Name of this section
                                               eSectionTypeCode,             // This section is a container of other sections.
                                               m_sect_headers[idx].vmaddr,   // File VM address == addresses as they are found in the object file
                                               m_sect_headers[idx].vmsize,   // VM size in bytes of this section
                                               m_sect_headers[idx].offset,   // Offset to the data for this section in the file
                                               m_sect_headers[idx].size,     // Size in bytes of this section as found in the the file
                                               m_sect_headers[idx].flags));  // Flags for this section

            //section_sp->SetIsEncrypted (segment_is_encrypted);

            m_sections_ap->AddSection(section_sp);
        }
    }
    return m_sections_ap.get();
}

bool
ObjectFilePECOFF::GetUUID (UUID* uuid)
{
    return false;
}

uint32_t
ObjectFilePECOFF::GetDependentModules (FileSpecList& files)
{
    return 0;
}


//----------------------------------------------------------------------
// Dump
//
// Dump the specifics of the runtime file container (such as any headers
// segments, sections, etc).
//----------------------------------------------------------------------
void 
ObjectFilePECOFF::Dump(Stream *s)
{
    Mutex::Locker locker(m_mutex);
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    s->PutCString("ObjectFilePECOFF");
    
    ArchSpec header_arch;
    GetArchitecture (header_arch);
    
    *s << ", file = '" << m_file << "', arch = " << header_arch.GetArchitectureName() << "\n";
    
    if (m_sections_ap.get())
        m_sections_ap->Dump(s, NULL, true, UINT32_MAX);
    
    if (m_symtab_ap.get())
        m_symtab_ap->Dump(s, NULL, eSortOrderNone);

    if (m_dos_header.e_magic)
        DumpDOSHeader (s, m_dos_header);
    if (m_coff_header.machine)
    {
        DumpCOFFHeader (s, m_coff_header);
        if (m_coff_header.hdrsize)
            DumpOptCOFFHeader (s, m_coff_header_opt);
    }
    s->EOL();
    DumpSectionHeaders(s);
    s->EOL();    
}

//----------------------------------------------------------------------
// DumpDOSHeader
//
// Dump the MS-DOS header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFilePECOFF::DumpDOSHeader(Stream *s, const dos_header_t& header)
{
    s->PutCString ("MSDOS Header\n");
    s->Printf ("  e_magic    = 0x%4.4x\n", header.e_magic);
    s->Printf ("  e_cblp     = 0x%4.4x\n", header.e_cblp);
    s->Printf ("  e_cp       = 0x%4.4x\n", header.e_cp);
    s->Printf ("  e_crlc     = 0x%4.4x\n", header.e_crlc);
    s->Printf ("  e_cparhdr  = 0x%4.4x\n", header.e_cparhdr);
    s->Printf ("  e_minalloc = 0x%4.4x\n", header.e_minalloc);
    s->Printf ("  e_maxalloc = 0x%4.4x\n", header.e_maxalloc);
    s->Printf ("  e_ss       = 0x%4.4x\n", header.e_ss);
    s->Printf ("  e_sp       = 0x%4.4x\n", header.e_sp);
    s->Printf ("  e_csum     = 0x%4.4x\n", header.e_csum);
    s->Printf ("  e_ip       = 0x%4.4x\n", header.e_ip);
    s->Printf ("  e_cs       = 0x%4.4x\n", header.e_cs);
    s->Printf ("  e_lfarlc   = 0x%4.4x\n", header.e_lfarlc);
    s->Printf ("  e_ovno     = 0x%4.4x\n", header.e_ovno);
    s->Printf ("  e_res[4]   = { 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x }\n",
               header.e_res[0],
               header.e_res[1],
               header.e_res[2],
               header.e_res[3]);
    s->Printf ("  e_oemid    = 0x%4.4x\n", header.e_oemid);
    s->Printf ("  e_oeminfo  = 0x%4.4x\n", header.e_oeminfo);
    s->Printf ("  e_res2[10] = { 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x }\n",
               header.e_res2[0],
               header.e_res2[1],
               header.e_res2[2],
               header.e_res2[3],
               header.e_res2[4],
               header.e_res2[5],
               header.e_res2[6],
               header.e_res2[7],
               header.e_res2[8],
               header.e_res2[9]);
    s->Printf ("  e_lfanew   = 0x%8.8x\n", header.e_lfanew);
}

//----------------------------------------------------------------------
// DumpCOFFHeader
//
// Dump the COFF header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFilePECOFF::DumpCOFFHeader(Stream *s, const coff_header_t& header)
{
    s->PutCString ("COFF Header\n");
    s->Printf ("  machine = 0x%4.4x\n", header.machine);
    s->Printf ("  nsects  = 0x%4.4x\n", header.nsects);
    s->Printf ("  modtime = 0x%8.8x\n", header.modtime);
    s->Printf ("  symoff  = 0x%8.8x\n", header.symoff);
    s->Printf ("  nsyms   = 0x%8.8x\n", header.nsyms);
    s->Printf ("  hdrsize = 0x%4.4x\n", header.hdrsize);
}

//----------------------------------------------------------------------
// DumpOptCOFFHeader
//
// Dump the optional COFF header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFilePECOFF::DumpOptCOFFHeader(Stream *s, const coff_opt_header_t& header)
{
    s->PutCString ("Optional COFF Header\n");
    s->Printf ("  magic                   = 0x%4.4x\n", header.magic);
    s->Printf ("  major_linker_version    = 0x%2.2x\n", header.major_linker_version);
    s->Printf ("  minor_linker_version    = 0x%2.2x\n", header.minor_linker_version);
    s->Printf ("  code_size               = 0x%8.8x\n", header.code_size);
    s->Printf ("  data_size               = 0x%8.8x\n", header.data_size);
    s->Printf ("  bss_size                = 0x%8.8x\n", header.bss_size);
    s->Printf ("  entry                   = 0x%8.8x\n", header.entry);
    s->Printf ("  code_offset             = 0x%8.8x\n", header.code_offset);
    s->Printf ("  data_offset             = 0x%8.8x\n", header.data_offset);
    s->Printf ("  image_base              = 0x%16.16llx\n", header.image_base);
    s->Printf ("  sect_alignment          = 0x%8.8x\n", header.sect_alignment);
    s->Printf ("  file_alignment          = 0x%8.8x\n", header.file_alignment);
    s->Printf ("  major_os_system_version = 0x%4.4x\n", header.major_os_system_version);
    s->Printf ("  minor_os_system_version = 0x%4.4x\n", header.minor_os_system_version);
    s->Printf ("  major_image_version     = 0x%4.4x\n", header.major_image_version);
    s->Printf ("  minor_image_version     = 0x%4.4x\n", header.minor_image_version);
    s->Printf ("  major_subsystem_version = 0x%4.4x\n", header.major_subsystem_version);
    s->Printf ("  minor_subsystem_version = 0x%4.4x\n", header.minor_subsystem_version);
    s->Printf ("  reserved1               = 0x%8.8x\n", header.reserved1);
    s->Printf ("  image_size              = 0x%8.8x\n", header.image_size);
    s->Printf ("  header_size             = 0x%8.8x\n", header.header_size);
    s->Printf ("  checksym                = 0x%8.8x\n", header.checksym);
    s->Printf ("  subsystem               = 0x%4.4x\n", header.subsystem);
    s->Printf ("  dll_flags               = 0x%4.4x\n", header.dll_flags);
    s->Printf ("  stack_reserve_size      = 0x%16.16llx\n", header.stack_reserve_size);
    s->Printf ("  stack_commit_size       = 0x%16.16llx\n", header.stack_commit_size);
    s->Printf ("  heap_reserve_size       = 0x%16.16llx\n", header.heap_reserve_size);
    s->Printf ("  heap_commit_size        = 0x%16.16llx\n", header.heap_commit_size);
    s->Printf ("  loader_flags            = 0x%8.8x\n", header.loader_flags);
    s->Printf ("  num_data_dir_entries    = 0x%8.8zx\n", header.data_dirs.size());
    uint32_t i;
    for (i=0; i<header.data_dirs.size(); i++)
    {
        s->Printf ("  data_dirs[%u] vmaddr = 0x%8.8x, vmsize = 0x%8.8x\n", 
                   i,
                   header.data_dirs[i].vmaddr,
                   header.data_dirs[i].vmsize);
    }
}
//----------------------------------------------------------------------
// DumpSectionHeader
//
// Dump a single ELF section header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFilePECOFF::DumpSectionHeader(Stream *s, const section_header_t& sh)
{
    std::string name;
    GetSectionName(name, sh);
    s->Printf ("%-16s 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x 0x%4.4x 0x%4.4x 0x%8.8x\n",
               name.c_str(),
               sh.vmsize,
               sh.vmaddr,
               sh.size,
               sh.offset,
               sh.reloff,
               sh.lineoff,
               sh.nreloc,
               sh.nline,
               sh.flags);
}


//----------------------------------------------------------------------
// DumpSectionHeaders
//
// Dump all of the ELF section header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFilePECOFF::DumpSectionHeaders(Stream *s)
{
    
    s->PutCString ("Section Headers\n");
    s->PutCString ("IDX  name             vm size  vm addr  size     offset   reloff   lineoff  nrel nlin flags\n");
    s->PutCString ("==== ---------------- -------- -------- -------- -------- -------- -------- ---- ---- --------\n");
    
    uint32_t idx = 0;
    SectionHeaderCollIter pos, end = m_sect_headers.end();
    
    for (pos = m_sect_headers.begin(); pos != end; ++pos, ++idx)
    {
        s->Printf ("[%2u]", idx);
        ObjectFilePECOFF::DumpSectionHeader(s, *pos);
    }
}

static bool 
COFFMachineToMachCPU (uint16_t machine, ArchSpec &arch)
{
    switch (machine)
    {
        case IMAGE_FILE_MACHINE_AMD64:
        case IMAGE_FILE_MACHINE_IA64:
            arch.SetArchitecture (eArchTypeMachO, 
                                  llvm::MachO::CPUTypeX86_64,
                                  llvm::MachO::CPUSubType_X86_64_ALL);
            return true;

        case IMAGE_FILE_MACHINE_I386:
            arch.SetArchitecture (eArchTypeMachO, 
                                  llvm::MachO::CPUTypeI386,
                                  llvm::MachO::CPUSubType_I386_ALL);
            return true;
            
        case IMAGE_FILE_MACHINE_POWERPC:    
        case IMAGE_FILE_MACHINE_POWERPCFP:  
            arch.SetArchitecture (eArchTypeMachO, 
                                  llvm::MachO::CPUTypePowerPC,
                                  llvm::MachO::CPUSubType_POWERPC_ALL);
            return true;
        case IMAGE_FILE_MACHINE_ARM:
        case IMAGE_FILE_MACHINE_THUMB:
            arch.SetArchitecture (eArchTypeMachO, 
                                  llvm::MachO::CPUTypeARM,
                                  llvm::MachO::CPUSubType_ARM_V7);
            return true;
    }
    return false;
}
bool
ObjectFilePECOFF::GetArchitecture (ArchSpec &arch)
{
    // For index zero return our cpu type
    return COFFMachineToMachCPU (m_coff_header.machine, arch);
}

ObjectFile::Type
ObjectFilePECOFF::CalculateType()
{
    if (m_coff_header.machine != 0)
    {
        if ((m_coff_header.flags & IMAGE_FILE_DLL) == 0)
            return eTypeExecutable;
        else
            return eTypeSharedLibrary;
    }
    return eTypeExecutable;
}

ObjectFile::Strata
ObjectFilePECOFF::CalculateStrata()
{
    return eStrataUser;
}
//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ObjectFilePECOFF::GetPluginName()
{
    return "ObjectFilePECOFF";
}

const char *
ObjectFilePECOFF::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjectFilePECOFF::GetPluginVersion()
{
    return 1;
}

