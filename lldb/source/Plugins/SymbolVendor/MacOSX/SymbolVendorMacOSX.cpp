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
SymbolVendorMacOSX::SymbolVendorMacOSX(Module *module) :
    SymbolVendor(module)
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
SymbolVendorMacOSX::CreateInstance(Module* module)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolVendorMacOSX::CreateInstance (module = %s/%s)",
                        module->GetFileSpec().GetDirectory().AsCString(),
                        module->GetFileSpec().GetFilename().AsCString());
    SymbolVendorMacOSX* symbol_vendor = new SymbolVendorMacOSX(module);
    if (symbol_vendor)
    {
        char path[PATH_MAX];
        path[0] = '\0';

        // Try and locate the dSYM file on Mac OS X
        ObjectFile * obj_file = module->GetObjectFile();
        if (obj_file)
        {
            Timer scoped_timer2 ("SymbolVendorMacOSX::CreateInstance () locate dSYM",
                                 "SymbolVendorMacOSX::CreateInstance (module = %s/%s) locate dSYM",
                                 module->GetFileSpec().GetDirectory().AsCString(),
                                 module->GetFileSpec().GetFilename().AsCString());

            FileSpec dsym_fspec;
            ObjectFileSP dsym_objfile_sp;
            const FileSpec &file_spec = obj_file->GetFileSpec();
            if (file_spec)
            {
                dsym_fspec = Symbols::LocateExecutableSymbolFile (&file_spec, &module->GetArchitecture(), &module->GetUUID());

                if (dsym_fspec)
                {
                    dsym_objfile_sp = ObjectFile::FindPlugin(module, &dsym_fspec, 0, dsym_fspec.GetByteSize());
                    if (UUIDsMatch(module, dsym_objfile_sp.get()))
                    {
                        ReplaceDSYMSectionsWithExecutableSections (obj_file, dsym_objfile_sp.get());
                        symbol_vendor->AddSymbolFileRepresendation(dsym_objfile_sp);
                        return symbol_vendor;
                    }
                }
            }

            // Just create our symbol vendor using the current objfile as this is either
            // an executable with no dSYM (that we could locate), an executable with
            // a dSYM that has a UUID that doesn't match.
            symbol_vendor->AddSymbolFileRepresendation(obj_file->GetSP());
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

