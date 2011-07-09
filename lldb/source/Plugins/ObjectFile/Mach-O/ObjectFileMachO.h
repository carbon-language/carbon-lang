//===-- ObjectFileMachO.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjectFileMachO_h_
#define liblldb_ObjectFileMachO_h_

#include "llvm/Support/MachO.h"

#include "lldb/Core/Address.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Symbol/ObjectFile.h"

//----------------------------------------------------------------------
// This class needs to be hidden as eventually belongs in a plugin that
// will export the ObjectFile protocol
//----------------------------------------------------------------------
class ObjectFileMachO :
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

    static ObjectFile *
    CreateInstance (lldb_private::Module* module,
                    lldb::DataBufferSP& dataSP,
                    const lldb_private::FileSpec* file,
                    lldb::addr_t offset,
                    lldb::addr_t length);

    static bool
    MagicBytesMatch (lldb::DataBufferSP& dataSP);

    //------------------------------------------------------------------
    // Member Functions
    //------------------------------------------------------------------
    ObjectFileMachO (lldb_private::Module* module,
                     lldb::DataBufferSP& dataSP,
                     const lldb_private::FileSpec* file,
                     lldb::addr_t offset,
                     lldb::addr_t length);

    virtual
    ~ObjectFileMachO();

    virtual bool
    ParseHeader ();

    virtual lldb::ByteOrder
    GetByteOrder () const;
    
    virtual bool
    IsExecutable () const;

    virtual size_t
    GetAddressByteSize ()  const;

    virtual lldb_private::AddressClass
    GetAddressClass (lldb::addr_t file_addr);

    virtual lldb_private::Symtab *
    GetSymtab();

    virtual lldb_private::SectionList *
    GetSectionList();

    virtual void
    Dump (lldb_private::Stream *s);

    virtual bool
    GetArchitecture (lldb_private::ArchSpec &arch);

    virtual bool
    GetUUID (lldb_private::UUID* uuid);

    virtual uint32_t
    GetDependentModules (lldb_private::FileSpecList& files);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();

    virtual const char *
    GetShortPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual lldb_private::Address
    GetEntryPointAddress ();

    virtual ObjectFile::Type
    CalculateType();
    
    virtual ObjectFile::Strata
    CalculateStrata();

protected:
    mutable lldb_private::Mutex m_mutex;
    llvm::MachO::mach_header m_header;
    mutable std::auto_ptr<lldb_private::SectionList> m_sections_ap;
    mutable std::auto_ptr<lldb_private::Symtab> m_symtab_ap;

    llvm::MachO::dysymtab_command m_dysymtab;
    std::vector<llvm::MachO::segment_command_64> m_mach_segments;
    std::vector<llvm::MachO::section_64> m_mach_sections;
    lldb_private::Address  m_entry_point_address;

    size_t
    ParseSections ();

    size_t
    ParseSymtab (bool minimize);

};

#endif  // liblldb_ObjectFileMachO_h_
