//===-- ObjectFileELF64.h ------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjectFileELF64_h_
#define liblldb_ObjectFileELF64_h_

#include <stdint.h>
#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Symbol/ObjectFile.h"

#include "elf.h"

//----------------------------------------------------------------------
// This class needs to be hidden as eventually belongs in a plugin that
// will export the ObjectFile protocol
//----------------------------------------------------------------------
class ObjectFileELF64 : 
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

    static const char *
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::ObjectFile *
    CreateInstance(lldb_private::Module* module,
                   lldb::DataBufferSP& dataSP,
                   const lldb_private::FileSpec* file,
                   lldb::addr_t offset,
                   lldb::addr_t length);

    static bool
    MagicBytesMatch(lldb::DataBufferSP& dataSP);

    //------------------------------------------------------------------
    // Member Functions
    //------------------------------------------------------------------
    ObjectFileELF64(lldb_private::Module* module,
                    lldb::DataBufferSP& dataSP,
                    const lldb_private::FileSpec* file,
                    lldb::addr_t offset,
                    lldb::addr_t length);

    virtual
    ~ObjectFileELF64();

    virtual bool
    ParseHeader();

    virtual lldb::ByteOrder
    GetByteOrder() const;

    virtual size_t
    GetAddressByteSize() const;

    virtual lldb_private::Symtab *
    GetSymtab();

    virtual lldb_private::SectionList *
    GetSectionList();

    virtual void
    Dump(lldb_private::Stream *s);

    virtual bool
    GetTargetTriple(lldb_private::ConstString &target_triple);

    virtual bool
    GetUUID(lldb_private::UUID* uuid);

    virtual uint32_t
    GetDependentModules(lldb_private::FileSpecList& files);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();

    virtual const char *
    GetShortPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual void
    GetPluginCommandHelp(const char *command, lldb_private::Stream *strm);

    virtual lldb_private::Error
    ExecutePluginCommand(lldb_private::Args &command,
                         lldb_private::Stream *strm);

    virtual lldb_private::Log *
    EnablePluginLogging(lldb_private::Stream *strm, 
                        lldb_private::Args &command);

protected:
    typedef std::vector<Elf64_Phdr>             ProgramHeaderColl;
    typedef ProgramHeaderColl::iterator         ProgramHeaderCollIter;
    typedef ProgramHeaderColl::const_iterator   ProgramHeaderCollConstIter;

    typedef std::vector<Elf64_Shdr>             SectionHeaderColl;
    typedef SectionHeaderColl::iterator         SectionHeaderCollIter;
    typedef SectionHeaderColl::const_iterator   SectionHeaderCollConstIter;

    Elf64_Ehdr m_header;
    ProgramHeaderColl m_program_headers;
    SectionHeaderColl m_section_headers;
    mutable std::auto_ptr<lldb_private::SectionList> m_sections_ap;
    mutable std::auto_ptr<lldb_private::Symtab> m_symtab_ap;
    mutable std::auto_ptr<lldb_private::FileSpecList> m_filespec_ap;
    lldb_private::DataExtractor m_shstr_data;

    size_t
    ParseSections();

    size_t
    ParseSymtab(bool minimize);

private:
    // Returns the 1 based index of a section header.
    unsigned
    SectionIndex(const SectionHeaderCollIter &I);

    unsigned
    SectionIndex(const SectionHeaderCollConstIter &I) const;

    // ELF header dump routines
    static void
    DumpELFHeader(lldb_private::Stream *s, const Elf64_Ehdr& header);

    static void
    DumpELFHeader_e_ident_EI_DATA(lldb_private::Stream *s, 
                                  unsigned char ei_data);

    static void
    DumpELFHeader_e_type(lldb_private::Stream *s, Elf64_Half e_type);

    // ELF program header dump routines
    void
    DumpELFProgramHeaders(lldb_private::Stream *s);

    static void
    DumpELFProgramHeader(lldb_private::Stream *s, const Elf64_Phdr &ph);

    static void
    DumpELFProgramHeader_p_type(lldb_private::Stream *s, Elf64_Word p_type);

    static void
    DumpELFProgramHeader_p_flags(lldb_private::Stream *s, Elf64_Word p_flags);

    // ELF section header dump routines
    void
    DumpELFSectionHeaders(lldb_private::Stream *s);

    static void
    DumpELFSectionHeader(lldb_private::Stream *s, const Elf64_Shdr& sh);

    static void
    DumpELFSectionHeader_sh_type(lldb_private::Stream *s, Elf64_Word sh_type);

    static void
    DumpELFSectionHeader_sh_flags(lldb_private::Stream *s, Elf64_Word sh_flags);

    void
    DumpDependentModules(lldb_private::Stream *s);

    size_t
    ParseProgramHeaders();

    size_t
    ParseSectionHeaders();

    size_t
    ParseDependentModules();

    void
    ParseSymbolTable(lldb_private::Symtab *symbol_table, 
                     const Elf64_Shdr &symtab_section,
                     lldb::user_id_t symtab_id);

    size_t
    GetSectionHeaderStringTable();

    uint32_t
    GetSectionIndexByName(const char *name);
};

#endif // #ifndef liblldb_ObjectFileELF64_h_
