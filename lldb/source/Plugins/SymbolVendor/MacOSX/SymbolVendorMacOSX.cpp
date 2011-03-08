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


//ObjectFile *
//LocateDSYMMachFileInDSYMBundle (Module* module, FileSpec& dsym_fspec)
//{
//    ObjectFile *dsym_objfile = NULL;
//
//    char path[PATH_MAX];
//
//    if (dsym_fspec.GetPath(path, sizeof(path)))
//    {
//        size_t path_len = strlen(path);
//        const char *bundle_subpath = "/Contents/Resources/DWARF/";
//        if (path_len > 0)
//        {
//            if (path[path_len-1] == '/')
//                ::strncat (path, bundle_subpath + 1, sizeof(path));
//            else
//                ::strncat (path, bundle_subpath, sizeof(path));
//            ::strncat (path, dsym_fspec.GetFilename().AsCString(), sizeof(path));
//
//            path_len = strlen(path);
//
//            if (::strcasecmp (&path[path_len - strlen(".dSYM")], ".dSYM") == 0)
//            {
//                path[path_len - ::strlen(".dSYM")] = '\0';
//                dsym_fspec.SetFile(path);
//                dsym_objfile = ObjectFile::FindPlugin(module, &dsym_fspec, 0);
//            }
//        }
//    }
//    return dsym_objfile;
//}
//
//CFURLRef DBGCopyFullDSYMURLForUUID (CFUUIDRef uuid, CFURLRef exec_url) __attribute__((weak_import));


//ObjectFile *
//FindDSYMUsingDebugSymbols (Module* module, FileSpec& dsym_fspec)
//{
//    Timer scoped_locate("FindDSYMUsingDebugSymbols");
//    dsym_fspec.Clear();
//    ObjectFile *dsym_objfile = NULL;
//    if (module->GetUUID().IsValid())
//    {
//        // Try and locate the dSYM file using DebugSymbols first
//        const UInt8 *module_uuid = (const UInt8 *)module->GetUUID().GetBytes();
//        if (module_uuid != NULL)
//        {
//            CFUUIDRef module_uuid_ref;
//            module_uuid_ref = ::CFUUIDCreateWithBytes ( NULL,
//                                                        module_uuid[0],
//                                                        module_uuid[1],
//                                                        module_uuid[2],
//                                                        module_uuid[3],
//                                                        module_uuid[4],
//                                                        module_uuid[5],
//                                                        module_uuid[6],
//                                                        module_uuid[7],
//                                                        module_uuid[8],
//                                                        module_uuid[9],
//                                                        module_uuid[10],
//                                                        module_uuid[11],
//                                                        module_uuid[12],
//                                                        module_uuid[13],
//                                                        module_uuid[14],
//                                                        module_uuid[15]);
//
//            if (module_uuid_ref)
//            {
//                CFURLRef dsym_url = NULL;
//                CFURLRef exec_url = NULL;
//
//            //  if (DBGCopyFullDSYMURLForUUID)
//                {
//                    char exec_path[PATH_MAX];
//                    if (module->GetFileSpec().GetPath(exec_path, sizeof(exec_path)))
//                    {
//                        exec_url = CFURLCreateFromFileSystemRepresentation ( NULL,
//                                                                             (const UInt8 *)exec_path,
//                                                                             strlen(exec_path),
//                                                                             FALSE);
//                    }
//
//                    dsym_url = DBGCopyFullDSYMURLForUUID(module_uuid_ref, exec_url);
//                }
//    //          else
//    //          {
//    //              dsym_url = DBGCopyDSYMURLForUUID(module_uuid_ref);
//    //          }
//
//                if (exec_url)
//                {
//                    ::CFRelease (exec_url);
//                    exec_url = NULL;
//                }
//
//                ::CFRelease(module_uuid_ref);
//                module_uuid_ref = NULL;
//
//                if (dsym_url)
//                {
//                    char dsym_path[PATH_MAX];
//                    Boolean success = CFURLGetFileSystemRepresentation (dsym_url, true, (UInt8*)dsym_path, sizeof(dsym_path)-1);
//
//                    ::CFRelease(dsym_url), dsym_url = NULL;
//
//                    if (success)
//                    {
//                        dsym_fspec.SetFile(dsym_path);
//
//                        // Some newer versions of DebugSymbols will return a full path into a dSYM bundle
//                        // that points to the correct mach file within the dSYM bundle (MH_DSYM mach file
//                        // type).
//                        dsym_objfile = ObjectFile::FindPlugin(module, &dsym_fspec, 0);
//
//                        // Olders versions of DebugSymbols will return a path to a dSYM bundle.
//                        if (dsym_objfile == NULL)
//                            dsym_objfile = LocateDSYMMachFileInDSYMBundle (module, dsym_fspec);
//                    }
//                }
//            }
//        }
//    }
//    return dsym_objfile;
//}

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
            std::auto_ptr<ObjectFile> dsym_objfile_ap;
            const FileSpec &file_spec = obj_file->GetFileSpec();
            if (file_spec)
            {
                dsym_fspec = Symbols::LocateExecutableSymbolFile (&file_spec, &module->GetArchitecture(), &module->GetUUID());

                if (dsym_fspec)
                {
                    dsym_objfile_ap.reset(ObjectFile::FindPlugin(module, &dsym_fspec, 0, dsym_fspec.GetByteSize()));
                    if (UUIDsMatch(module, dsym_objfile_ap.get()))
                    {
                        ReplaceDSYMSectionsWithExecutableSections (obj_file, dsym_objfile_ap.get());
                        symbol_vendor->AddSymbolFileRepresendation(dsym_objfile_ap.release());
                        return symbol_vendor;
                    }
                }
            }

            // Just create our symbol vendor using the current objfile as this is either
            // an executable with no dSYM (that we could locate), and executable with
            // a dSYM that has a UUID that doesn't match, or it is a dSYM file itself.
            symbol_vendor->AddSymbolFileRepresendation(obj_file);
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

