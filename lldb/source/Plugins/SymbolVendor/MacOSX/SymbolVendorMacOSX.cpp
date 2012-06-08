//===-- SymbolVendorMacOSX.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolVendorMacOSX.h"

#include <AvailabilityMacros.h>

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Symbols.h"
#include "lldb/Symbol/ObjectFile.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// SymbolVendorMacOSX constructor
//----------------------------------------------------------------------
SymbolVendorMacOSX::SymbolVendorMacOSX(const lldb::ModuleSP &module_sp) :
    SymbolVendor (module_sp)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SymbolVendorMacOSX::~SymbolVendorMacOSX()
{
}


static bool
UUIDsMatch(Module *module, ObjectFile *ofile)
{
    if (module && ofile)
    {
        // Make sure the UUIDs match
        lldb_private::UUID dsym_uuid;
        if (ofile->GetUUID(&dsym_uuid))
            return dsym_uuid == module->GetUUID();
    }
    return false;
}

static void
ReplaceDSYMSectionsWithExecutableSections (ObjectFile *exec_objfile, ObjectFile *dsym_objfile)
{
    // We need both the executable and the dSYM to live off of the
    // same section lists. So we take all of the sections from the
    // executable, and replace them in the dSYM. This allows section
    // offset addresses that come from the dSYM to automatically
    // get updated as images (shared libraries) get loaded and
    // unloaded.
    SectionList *exec_section_list = exec_objfile->GetSectionList();
    SectionList *dsym_section_list = dsym_objfile->GetSectionList();
    if (exec_section_list && dsym_section_list)
    {
        const uint32_t num_exec_sections = dsym_section_list->GetSize();
        uint32_t exec_sect_idx;
        for (exec_sect_idx = 0; exec_sect_idx < num_exec_sections; ++exec_sect_idx)
        {
            SectionSP exec_sect_sp(exec_section_list->GetSectionAtIndex(exec_sect_idx));
            if (exec_sect_sp.get())
            {
                // Try and replace any sections that exist in both the executable
                // and in the dSYM with those from the executable. If we fail to
                // replace the one in the dSYM, then add the executable section to
                // the dSYM.
                if (dsym_section_list->ReplaceSection(exec_sect_sp->GetID(), exec_sect_sp, 0) == false)
                    dsym_section_list->AddSection(exec_sect_sp);
            }
        }
        
        dsym_section_list->Finalize(); // Now that we're done adding sections, finalize to build fast-lookup caches
    }
}

void
SymbolVendorMacOSX::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
SymbolVendorMacOSX::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
SymbolVendorMacOSX::GetPluginNameStatic()
{
    return "symbol-vendor.macosx";
}

const char *
SymbolVendorMacOSX::GetPluginDescriptionStatic()
{
    return "Symbol vendor for MacOSX that looks for dSYM files that match executables.";
}



//----------------------------------------------------------------------
// CreateInstance
//
// Platforms can register a callback to use when creating symbol
// vendors to allow for complex debug information file setups, and to
// also allow for finding separate debug information files.
//----------------------------------------------------------------------
SymbolVendor*
SymbolVendorMacOSX::CreateInstance (const lldb::ModuleSP &module_sp)
{
    if (!module_sp)
        return NULL;

    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolVendorMacOSX::CreateInstance (module = %s/%s)",
                        module_sp->GetFileSpec().GetDirectory().AsCString(),
                        module_sp->GetFileSpec().GetFilename().AsCString());
    SymbolVendorMacOSX* symbol_vendor = new SymbolVendorMacOSX(module_sp);
    if (symbol_vendor)
    {
        char path[PATH_MAX];
        path[0] = '\0';

        // Try and locate the dSYM file on Mac OS X
        ObjectFile * obj_file = module_sp->GetObjectFile();
        if (obj_file)
        {
            Timer scoped_timer2 ("SymbolVendorMacOSX::CreateInstance () locate dSYM",
                                 "SymbolVendorMacOSX::CreateInstance (module = %s/%s) locate dSYM",
                                 module_sp->GetFileSpec().GetDirectory().AsCString(),
                                 module_sp->GetFileSpec().GetFilename().AsCString());

            // First check to see if the module has a symbol file in mind already.
            // If it does, then we MUST use that.
            FileSpec dsym_fspec (module_sp->GetSymbolFileFileSpec());
            
            ObjectFileSP dsym_objfile_sp;
            if (!dsym_fspec)
            {
                // No symbol file was specified in the module, lets try and find
                // one ourselves.
                const FileSpec &file_spec = obj_file->GetFileSpec();
                if (file_spec)
                {
                    ModuleSpec module_spec(file_spec, module_sp->GetArchitecture());
                    module_spec.GetUUID() = module_sp->GetUUID();
                    dsym_fspec = Symbols::LocateExecutableSymbolFile (module_spec);
                    if (module_spec.GetSourceMappingList().GetSize())
                    {
                        module_sp->GetSourceMappingList().Append (module_spec.GetSourceMappingList (), true);
                    }
                }
            }
            
            if (dsym_fspec)
            {
                DataBufferSP dsym_file_data_sp;
                dsym_objfile_sp = ObjectFile::FindPlugin(module_sp, &dsym_fspec, 0, dsym_fspec.GetByteSize(), dsym_file_data_sp);
                if (UUIDsMatch(module_sp.get(), dsym_objfile_sp.get()))
                {
                    ReplaceDSYMSectionsWithExecutableSections (obj_file, dsym_objfile_sp.get());
                    symbol_vendor->AddSymbolFileRepresentation(dsym_objfile_sp);
                    return symbol_vendor;
                }
            }

            // Just create our symbol vendor using the current objfile as this is either
            // an executable with no dSYM (that we could locate), an executable with
            // a dSYM that has a UUID that doesn't match.
            symbol_vendor->AddSymbolFileRepresentation(obj_file->shared_from_this());
        }
    }
    return symbol_vendor;
}



//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
SymbolVendorMacOSX::GetPluginName()
{
    return "SymbolVendorMacOSX";
}

const char *
SymbolVendorMacOSX::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
SymbolVendorMacOSX::GetPluginVersion()
{
    return 1;
}

