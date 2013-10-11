//===-- ObjectFileELF.h --------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjectFileELF_h_
#define liblldb_ObjectFileELF_h_

#include <stdint.h>
#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Core/UUID.h"

#include "ELFHeader.h"

//------------------------------------------------------------------------------
/// @class ObjectFileELF
/// @brief Generic ELF object file reader.
///
/// This class provides a generic ELF (32/64 bit) reader plugin implementing the
/// ObjectFile protocol.
class ObjectFileELF :
    public lldb_private::ObjectFile
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::ObjectFile *
    CreateInstance(const lldb::ModuleSP &module_sp,
                   lldb::DataBufferSP& data_sp,
                   lldb::offset_t data_offset,
                   const lldb_private::FileSpec* file,
                   lldb::offset_t file_offset,
                   lldb::offset_t length);

    static lldb_private::ObjectFile *
    CreateMemoryInstance (const lldb::ModuleSP &module_sp, 
                          lldb::DataBufferSP& data_sp, 
                          const lldb::ProcessSP &process_sp, 
                          lldb::addr_t header_addr);

    static size_t
    GetModuleSpecifications (const lldb_private::FileSpec& file,
                             lldb::DataBufferSP& data_sp,
                             lldb::offset_t data_offset,
                             lldb::offset_t file_offset,
                             lldb::offset_t length,
                             lldb_private::ModuleSpecList &specs);

    static bool
    MagicBytesMatch (lldb::DataBufferSP& data_sp,
                     lldb::addr_t offset, 
                     lldb::addr_t length);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

    //------------------------------------------------------------------
    // ObjectFile Protocol.
    //------------------------------------------------------------------
    virtual
    ~ObjectFileELF();

    virtual bool
    ParseHeader();

    virtual lldb::ByteOrder
    GetByteOrder() const;

    virtual bool
    IsExecutable () const;

    virtual uint32_t
    GetAddressByteSize() const;

    virtual lldb_private::Symtab *
    GetSymtab();

    virtual lldb_private::Symbol *
    ResolveSymbolForAddress(const lldb_private::Address& so_addr, bool verify_unique);

    virtual bool
    IsStripped ();

    virtual void
    CreateSections (lldb_private::SectionList &unified_section_list);

    virtual void
    Dump(lldb_private::Stream *s);

    virtual bool
    GetArchitecture (lldb_private::ArchSpec &arch);

    virtual bool
    GetUUID(lldb_private::UUID* uuid);

    virtual lldb_private::FileSpecList
    GetDebugSymbolFilePaths();

    virtual uint32_t
    GetDependentModules(lldb_private::FileSpecList& files);

    virtual lldb_private::Address
    GetImageInfoAddress(lldb_private::Target *target);
    
    virtual lldb_private::Address
    GetEntryPointAddress ();
    
    virtual ObjectFile::Type
    CalculateType();
    
    virtual ObjectFile::Strata
    CalculateStrata();

    // Returns number of program headers found in the ELF file.
    size_t
    GetProgramHeaderCount();

    // Returns the program header with the given index.
    const elf::ELFProgramHeader *
    GetProgramHeaderByIndex(lldb::user_id_t id);

    // Returns segment data for the given index.
    lldb_private::DataExtractor
    GetSegmentDataByIndex(lldb::user_id_t id);

private:
    ObjectFileELF(const lldb::ModuleSP &module_sp,
                  lldb::DataBufferSP& data_sp,
                  lldb::offset_t data_offset,
                  const lldb_private::FileSpec* file,
                  lldb::offset_t offset,
                  lldb::offset_t length);

    typedef std::vector<elf::ELFProgramHeader>  ProgramHeaderColl;
    typedef ProgramHeaderColl::iterator         ProgramHeaderCollIter;
    typedef ProgramHeaderColl::const_iterator   ProgramHeaderCollConstIter;

    struct ELFSectionHeaderInfo : public elf::ELFSectionHeader
    {
        lldb_private::ConstString section_name;
    };
    typedef std::vector<ELFSectionHeaderInfo>   SectionHeaderColl;
    typedef SectionHeaderColl::iterator         SectionHeaderCollIter;
    typedef SectionHeaderColl::const_iterator   SectionHeaderCollConstIter;

    typedef std::vector<elf::ELFDynamic>        DynamicSymbolColl;
    typedef DynamicSymbolColl::iterator         DynamicSymbolCollIter;
    typedef DynamicSymbolColl::const_iterator   DynamicSymbolCollConstIter;

    /// Version of this reader common to all plugins based on this class.
    static const uint32_t m_plugin_version = 1;

    /// ELF file header.
    elf::ELFHeader m_header;

    /// ELF build ID.
    lldb_private::UUID m_uuid;

    /// ELF .gnu_debuglink file and crc data if available.
    std::string m_gnu_debuglink_file;
    uint32_t m_gnu_debuglink_crc;

    /// Collection of program headers.
    ProgramHeaderColl m_program_headers;

    /// Collection of section headers.
    SectionHeaderColl m_section_headers;

    /// Collection of symbols from the dynamic table.
    DynamicSymbolColl m_dynamic_symbols;

    /// List of file specifications corresponding to the modules (shared
    /// libraries) on which this object file depends.
    mutable std::unique_ptr<lldb_private::FileSpecList> m_filespec_ap;

    /// Cached value of the entry point for this module.
    lldb_private::Address  m_entry_point_address;

    /// Returns a 1 based index of the given section header.
    size_t
    SectionIndex(const SectionHeaderCollIter &I);

    /// Returns a 1 based index of the given section header.
    size_t
    SectionIndex(const SectionHeaderCollConstIter &I) const;

    /// Parses all section headers present in this object file and populates
    /// m_program_headers.  This method will compute the header list only once.
    /// Returns the number of headers parsed.
    size_t
    ParseProgramHeaders();

    /// Parses all section headers present in this object file and populates
    /// m_section_headers.  This method will compute the header list only once.
    /// Returns the number of headers parsed.
    size_t
    ParseSectionHeaders();

    /// Parses the elf section headers and returns the uuid, debug link name, crc.
    static size_t
    GetSectionHeaderInfo(SectionHeaderColl &section_headers,
                         lldb_private::DataExtractor &data,
                         const elf::ELFHeader &header,
                         lldb_private::UUID &uuid,
                         std::string &gnu_debuglink_file,
                         uint32_t &gnu_debuglink_crc);

    /// Scans the dynamic section and locates all dependent modules (shared
    /// libraries) populating m_filespec_ap.  This method will compute the
    /// dependent module list only once.  Returns the number of dependent
    /// modules parsed.
    size_t
    ParseDependentModules();

    /// Parses the dynamic symbol table and populates m_dynamic_symbols.  The
    /// vector retains the order as found in the object file.  Returns the
    /// number of dynamic symbols parsed.
    size_t
    ParseDynamicSymbols();

    /// Populates m_symtab_ap will all non-dynamic linker symbols.  This method
    /// will parse the symbols only once.  Returns the number of symbols parsed.
    unsigned
    ParseSymbolTable(lldb_private::Symtab *symbol_table,
                     lldb::user_id_t start_id,
                     lldb_private::Section *symtab);

    /// Helper routine for ParseSymbolTable().
    unsigned
    ParseSymbols(lldb_private::Symtab *symbol_table, 
                 lldb::user_id_t start_id,
                 lldb_private::SectionList *section_list,
                 const size_t num_symbols,
                 const lldb_private::DataExtractor &symtab_data,
                 const lldb_private::DataExtractor &strtab_data);

    /// Scans the relocation entries and adds a set of artificial symbols to the
    /// given symbol table for each PLT slot.  Returns the number of symbols
    /// added.
    unsigned
    ParseTrampolineSymbols(lldb_private::Symtab *symbol_table, 
                           lldb::user_id_t start_id,
                           const ELFSectionHeaderInfo *rela_hdr,
                           lldb::user_id_t section_id);

    /// Returns the section header with the given id or NULL.
    const ELFSectionHeaderInfo *
    GetSectionHeaderByIndex(lldb::user_id_t id);

    /// @name  ELF header dump routines
    //@{
    static void
    DumpELFHeader(lldb_private::Stream *s, const elf::ELFHeader& header);

    static void
    DumpELFHeader_e_ident_EI_DATA(lldb_private::Stream *s,
                                  unsigned char ei_data);

    static void
    DumpELFHeader_e_type(lldb_private::Stream *s, elf::elf_half e_type);
    //@}

    /// @name ELF program header dump routines
    //@{
    void
    DumpELFProgramHeaders(lldb_private::Stream *s);

    static void
    DumpELFProgramHeader(lldb_private::Stream *s, 
                         const elf::ELFProgramHeader &ph);

    static void
    DumpELFProgramHeader_p_type(lldb_private::Stream *s, elf::elf_word p_type);

    static void
    DumpELFProgramHeader_p_flags(lldb_private::Stream *s, 
                                 elf::elf_word p_flags);
    //@}

    /// @name ELF section header dump routines
    //@{
    void
    DumpELFSectionHeaders(lldb_private::Stream *s);

    static void
    DumpELFSectionHeader(lldb_private::Stream *s, 
                         const ELFSectionHeaderInfo& sh);

    static void
    DumpELFSectionHeader_sh_type(lldb_private::Stream *s, 
                                 elf::elf_word sh_type);

    static void
    DumpELFSectionHeader_sh_flags(lldb_private::Stream *s, 
                                  elf::elf_xword sh_flags);
    //@}

    /// ELF dependent module dump routine.
    void
    DumpDependentModules(lldb_private::Stream *s);

    const elf::ELFDynamic *
    FindDynamicSymbol(unsigned tag);
        
    unsigned
    PLTRelocationType();
};

#endif // #ifndef liblldb_ObjectFileELF_h_
