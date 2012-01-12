//===-- ObjectFile.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Timer.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/ObjectContainer.h"
#include "lldb/Symbol/SymbolFile.h"

using namespace lldb;
using namespace lldb_private;

ObjectFileSP
ObjectFile::FindPlugin (Module* module, const FileSpec* file, addr_t file_offset, addr_t file_size, DataBufferSP &file_data_sp)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "ObjectFile::FindPlugin (module = %s/%s, file = %p, file_offset = 0x%z8.8x, file_size = 0x%z8.8x)",
                        module->GetFileSpec().GetDirectory().AsCString(),
                        module->GetFileSpec().GetFilename().AsCString(),
                        file, file_offset, file_size);
    ObjectFileSP object_file_sp;

    if (module != NULL)
    {
        if (file)
        {
            // Memory map the entire file contents
            if (!file_data_sp)
            {
                assert (file_offset == 0);
                file_data_sp = file->MemoryMapFileContents(file_offset, file_size);
            }

            if (!file_data_sp || file_data_sp->GetByteSize() == 0)
            {
                // Check for archive file with format "/path/to/archive.a(object.o)"
                char path_with_object[PATH_MAX*2];
                module->GetFileSpec().GetPath(path_with_object, sizeof(path_with_object));

                RegularExpression g_object_regex("(.*)\\(([^\\)]+)\\)$");
                if (g_object_regex.Execute (path_with_object, 2))
                {
                    FileSpec archive_file;
                    std::string path;
                    std::string object;
                    if (g_object_regex.GetMatchAtIndex (path_with_object, 1, path) &&
                        g_object_regex.GetMatchAtIndex (path_with_object, 2, object))
                    {
                        archive_file.SetFile (path.c_str(), false);
                        file_size = archive_file.GetByteSize();
                        if (file_size > 0)
                        {
                            module->SetFileSpecAndObjectName (archive_file, ConstString(object.c_str()));
                            file_data_sp = archive_file.MemoryMapFileContents(file_offset, file_size);
                        }
                    }
                }
            }

            if (file_data_sp && file_data_sp->GetByteSize() > 0)
            {
                uint32_t idx;

                // Check if this is a normal object file by iterating through
                // all object file plugin instances.
                ObjectFileCreateInstance create_object_file_callback;
                for (idx = 0; (create_object_file_callback = PluginManager::GetObjectFileCreateCallbackAtIndex(idx)) != NULL; ++idx)
                {
                    object_file_sp.reset (create_object_file_callback(module, file_data_sp, file, file_offset, file_size));
                    if (object_file_sp.get())
                        return object_file_sp;
                }

                // Check if this is a object container by iterating through
                // all object container plugin instances and then trying to get
                // an object file from the container.
                ObjectContainerCreateInstance create_object_container_callback;
                for (idx = 0; (create_object_container_callback = PluginManager::GetObjectContainerCreateCallbackAtIndex(idx)) != NULL; ++idx)
                {
                    std::auto_ptr<ObjectContainer> object_container_ap(create_object_container_callback(module, file_data_sp, file, file_offset, file_size));

                    if (object_container_ap.get())
                        object_file_sp = object_container_ap->GetObjectFile(file);

                    if (object_file_sp.get())
                        return object_file_sp;
                }
            }
        }
    }
    // We didn't find it, so clear our shared pointer in case it
    // contains anything and return an empty shared pointer
    object_file_sp.reset();
    return object_file_sp;
}

ObjectFile::ObjectFile (Module* module, 
                        const FileSpec *file_spec_ptr, 
                        addr_t file_offset, 
                        addr_t file_size, 
                        DataBufferSP& file_data_sp) :
    ModuleChild (module),
    m_file (),  // This file could be different from the original module's file
    m_type (eTypeInvalid),
    m_strata (eStrataInvalid),
    m_offset (file_offset),
    m_length (file_size),
    m_data (),
    m_unwind_table (*this)
{    
    if (file_spec_ptr)
        m_file = *file_spec_ptr;
    if (file_data_sp)
        m_data.SetData (file_data_sp, file_offset, file_size);
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
    {
        if (m_file)
        {
            log->Printf ("%p ObjectFile::ObjectFile () module = %s/%s, file = %s/%s, offset = 0x%8.8llx, size = %llu\n",
                         this,
                         m_module->GetFileSpec().GetDirectory().AsCString(),
                         m_module->GetFileSpec().GetFilename().AsCString(),
                         m_file.GetDirectory().AsCString(),
                         m_file.GetFilename().AsCString(),
                         m_offset,
                         m_length);
        }
        else
        {
            log->Printf ("%p ObjectFile::ObjectFile () module = %s/%s, file = <NULL>, offset = 0x%8.8llx, size = %llu\n",
                         this,
                         m_module->GetFileSpec().GetDirectory().AsCString(),
                         m_module->GetFileSpec().GetFilename().AsCString(),
                         m_offset,
                         m_length);
        }
    }
}

ObjectFile::~ObjectFile()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
    {
        if (m_file)
        {
            log->Printf ("%p ObjectFile::~ObjectFile () module = %s/%s, file = %s/%s, offset = 0x%8.8llx, size = %llu\n",
                         this,
                         m_module->GetFileSpec().GetDirectory().AsCString(),
                         m_module->GetFileSpec().GetFilename().AsCString(),
                         m_file.GetDirectory().AsCString(),
                         m_file.GetFilename().AsCString(),
                         m_offset,
                         m_length);
        }
        else
        {
            log->Printf ("%p ObjectFile::~ObjectFile () module = %s/%s, file = <NULL>, offset = 0x%8.8llx, size = %llu\n",
                         this,
                         m_module->GetFileSpec().GetDirectory().AsCString(),
                         m_module->GetFileSpec().GetFilename().AsCString(),
                         m_offset,
                         m_length);
        }
    }
}

bool 
ObjectFile::SetModulesArchitecture (const ArchSpec &new_arch)
{
    return m_module->SetArchitecture (new_arch);
}

AddressClass
ObjectFile::GetAddressClass (addr_t file_addr)
{
    Symtab *symtab = GetSymtab();
    if (symtab)
    {
        Symbol *symbol = symtab->FindSymbolContainingFileAddress(file_addr);
        if (symbol)
        {
            const AddressRange *range_ptr = symbol->GetAddressRangePtr();
            if (range_ptr)
            {
                const Section *section = range_ptr->GetBaseAddress().GetSection();
                if (section)
                {
                    const SectionType section_type = section->GetType();
                    switch (section_type)
                    {
                    case eSectionTypeInvalid:               return eAddressClassUnknown;
                    case eSectionTypeCode:                  return eAddressClassCode;
                    case eSectionTypeContainer:             return eAddressClassUnknown;
                    case eSectionTypeData:
                    case eSectionTypeDataCString:
                    case eSectionTypeDataCStringPointers:
                    case eSectionTypeDataSymbolAddress:
                    case eSectionTypeData4:
                    case eSectionTypeData8:
                    case eSectionTypeData16:
                    case eSectionTypeDataPointers:
                    case eSectionTypeZeroFill:
                    case eSectionTypeDataObjCMessageRefs:
                    case eSectionTypeDataObjCCFStrings:
                        return eAddressClassData;
                    case eSectionTypeDebug:
                    case eSectionTypeDWARFDebugAbbrev:
                    case eSectionTypeDWARFDebugAranges:
                    case eSectionTypeDWARFDebugFrame:
                    case eSectionTypeDWARFDebugInfo:
                    case eSectionTypeDWARFDebugLine:
                    case eSectionTypeDWARFDebugLoc:
                    case eSectionTypeDWARFDebugMacInfo:
                    case eSectionTypeDWARFDebugPubNames:
                    case eSectionTypeDWARFDebugPubTypes:
                    case eSectionTypeDWARFDebugRanges:
                    case eSectionTypeDWARFDebugStr:
                    case eSectionTypeDWARFAppleNames:
                    case eSectionTypeDWARFAppleTypes:
                    case eSectionTypeDWARFAppleNamespaces:
                    case eSectionTypeDWARFAppleObjC:
                        return eAddressClassDebug;
                    case eSectionTypeEHFrame:               return eAddressClassRuntime;
                    case eSectionTypeOther:                 return eAddressClassUnknown;
                    }
                }
            }
            
            const SymbolType symbol_type = symbol->GetType();
            switch (symbol_type)
            {
            case eSymbolTypeAny:            return eAddressClassUnknown;
            case eSymbolTypeAbsolute:       return eAddressClassUnknown;
            case eSymbolTypeCode:           return eAddressClassCode;
            case eSymbolTypeTrampoline:     return eAddressClassCode;
            case eSymbolTypeData:           return eAddressClassData;
            case eSymbolTypeRuntime:        return eAddressClassRuntime;
            case eSymbolTypeException:      return eAddressClassRuntime;
            case eSymbolTypeSourceFile:     return eAddressClassDebug;
            case eSymbolTypeHeaderFile:     return eAddressClassDebug;
            case eSymbolTypeObjectFile:     return eAddressClassDebug;
            case eSymbolTypeCommonBlock:    return eAddressClassDebug;
            case eSymbolTypeBlock:          return eAddressClassDebug;
            case eSymbolTypeLocal:          return eAddressClassData;
            case eSymbolTypeParam:          return eAddressClassData;
            case eSymbolTypeVariable:       return eAddressClassData;
            case eSymbolTypeVariableType:   return eAddressClassDebug;
            case eSymbolTypeLineEntry:      return eAddressClassDebug;
            case eSymbolTypeLineHeader:     return eAddressClassDebug;
            case eSymbolTypeScopeBegin:     return eAddressClassDebug;
            case eSymbolTypeScopeEnd:       return eAddressClassDebug;
            case eSymbolTypeAdditional:     return eAddressClassUnknown;
            case eSymbolTypeCompiler:       return eAddressClassDebug;
            case eSymbolTypeInstrumentation:return eAddressClassDebug;
            case eSymbolTypeUndefined:      return eAddressClassUnknown;
            case eSymbolTypeObjCClass:      return eAddressClassRuntime;
            case eSymbolTypeObjCMetaClass:  return eAddressClassRuntime;
            case eSymbolTypeObjCIVar:       return eAddressClassRuntime;
            }
        }
    }
    return eAddressClassUnknown;
}

ObjectFileSP
ObjectFile::GetSP ()
{
    // This object contains an instrusive ref count base class so we can
    // easily make a shared pointer to this object
    return ObjectFileSP (this);
}

size_t
ObjectFile::GetData (off_t offset, size_t length, DataExtractor &data) const
{
    // The entire file has already been mmap'ed into m_data, so just copy from there
    // as the back mmap buffer will be shared with shared pointers.
    return data.SetData (m_data, offset, length);
}

size_t
ObjectFile::CopyData (off_t offset, size_t length, void *dst) const
{
    // The entire file has already been mmap'ed into m_data, so just copy from there
    return m_data.CopyByteOrderedData (offset, length, dst, length, lldb::endian::InlHostByteOrder());
}

