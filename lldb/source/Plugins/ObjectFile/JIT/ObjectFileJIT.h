//===-- ObjectFileJIT.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjectFileJIT_h_
#define liblldb_ObjectFileJIT_h_

#include "lldb/Core/Address.h"
#include "lldb/Symbol/ObjectFile.h"


//----------------------------------------------------------------------
// This class needs to be hidden as eventually belongs in a plugin that
// will export the ObjectFile protocol
//----------------------------------------------------------------------
class ObjectFileJIT :
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
    CreateInstance (const lldb::ModuleSP &module_sp,
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
    
    //------------------------------------------------------------------
    // Member Functions
    //------------------------------------------------------------------
    ObjectFileJIT (const lldb::ModuleSP &module_sp,
                   const lldb::ObjectFileJITDelegateSP &delegate_sp);
    
    virtual
    ~ObjectFileJIT();

    virtual bool
    ParseHeader ();

    virtual bool
    SetLoadAddress(lldb_private::Target &target,
                   lldb::addr_t value,
                   bool value_is_offset);
    
    virtual lldb::ByteOrder
    GetByteOrder () const;
    
    virtual bool
    IsExecutable () const;

    virtual uint32_t
    GetAddressByteSize ()  const;

    virtual lldb_private::Symtab *
    GetSymtab();

    virtual bool
    IsStripped ();
    
    virtual void
    CreateSections (lldb_private::SectionList &unified_section_list);

    virtual void
    Dump (lldb_private::Stream *s);

    virtual bool
    GetArchitecture (lldb_private::ArchSpec &arch);

    virtual bool
    GetUUID (lldb_private::UUID* uuid);

    virtual uint32_t
    GetDependentModules (lldb_private::FileSpecList& files);
    
    virtual size_t
    ReadSectionData (const lldb_private::Section *section,
                     lldb::offset_t section_offset,
                     void *dst,
                     size_t dst_len) const;
    virtual size_t
    ReadSectionData (const lldb_private::Section *section,
                     lldb_private::DataExtractor& section_data) const;
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual lldb_private::Address
    GetEntryPointAddress ();
    
    virtual lldb_private::Address
    GetHeaderAddress ();
    
    virtual ObjectFile::Type
    CalculateType();
    
    virtual ObjectFile::Strata
    CalculateStrata();
protected:
    lldb::ObjectFileJITDelegateWP m_delegate_wp;
};

#endif  // liblldb_ObjectFileJIT_h_
