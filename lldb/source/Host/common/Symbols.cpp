//===-- Symbols.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Symbols.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/UUID.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

#if defined (__linux__) || defined (__FreeBSD__)

FileSpec
Symbols::LocateExecutableObjectFile (const ModuleSpec &module_spec)
{
    // FIXME
    return FileSpec();
}

FileSpec
Symbols::LocateExecutableSymbolFile (const ModuleSpec &module_spec)
{
    const char *symbol_filename = module_spec.GetSymbolFileSpec().GetFilename().AsCString();
    if (!symbol_filename || !symbol_filename[0])
        return FileSpec();

    FileSpecList debug_file_search_paths (Target::GetDefaultDebugFileSearchPaths());

    // Add module directory.
    const ConstString &file_dir = module_spec.GetFileSpec().GetDirectory();
    debug_file_search_paths.AppendIfUnique (FileSpec(file_dir.AsCString("."), true));

    // Add current working directory.
    debug_file_search_paths.AppendIfUnique (FileSpec(".", true));

    // Add /usr/lib/debug directory.
    debug_file_search_paths.AppendIfUnique (FileSpec("/usr/lib/debug", true));

    std::string uuid_str;
    const UUID &module_uuid = module_spec.GetUUID();
    if (module_uuid.IsValid())
    {
        // Some debug files are stored in the .build-id directory like this:
        //   /usr/lib/debug/.build-id/ff/e7fe727889ad82bb153de2ad065b2189693315.debug
        uuid_str = module_uuid.GetAsString("");
        uuid_str.insert (2, 1, '/');
        uuid_str = uuid_str + ".debug";
    }

    // Get full path to our module. Needed to check debug files like this:
    //   /usr/lib/debug/usr/lib/libboost_date_time.so.1.46.1
    std::string module_filename = module_spec.GetFileSpec().GetPath();

    size_t num_directories = debug_file_search_paths.GetSize();
    for (size_t idx = 0; idx < num_directories; ++idx)
    {
        FileSpec dirspec = debug_file_search_paths.GetFileSpecAtIndex (idx);
        dirspec.ResolvePath();
        if (!dirspec.Exists() || !dirspec.IsDirectory())
            continue;

        std::vector<std::string> files;
        std::string dirname = dirspec.GetPath();

        files.push_back (dirname + "/" + symbol_filename);
        files.push_back (dirname + "/.debug/" + symbol_filename);
        files.push_back (dirname + "/.build-id/" + uuid_str);
        files.push_back (dirname + module_filename);

        const uint32_t num_files = files.size();
        for (size_t idx_file = 0; idx_file < num_files; ++idx_file)
        {
            const std::string &filename = files[idx_file];
            FileSpec file_spec (filename.c_str(), true);

            if (file_spec == module_spec.GetFileSpec())
                continue;

            if (file_spec.Exists())
            {
                lldb_private::ModuleSpecList specs;
                const size_t num_specs = ObjectFile::GetModuleSpecifications (file_spec, 0, 0, specs);
                assert (num_specs <= 1 && "Symbol Vendor supports only a single architecture");
                if (num_specs == 1)
                {
                    ModuleSpec mspec;
                    if (specs.GetModuleSpecAtIndex (0, mspec))
                    {
                        if (mspec.GetUUID() == module_uuid)
                            return file_spec;
                    }
                }
            }
        }
    }

    return FileSpec();
}

FileSpec
Symbols::FindSymbolFileInBundle (const FileSpec& symfile_bundle,
                                 const lldb_private::UUID *uuid,
                                 const ArchSpec *arch)
{
    // FIXME
    return FileSpec();
}

bool
Symbols::DownloadObjectAndSymbolFile (ModuleSpec &module_spec, bool force_lookup)
{
    // Fill in the module_spec.GetFileSpec() for the object file and/or the
    // module_spec.GetSymbolFileSpec() for the debug symbols file.
    return false;
}

#elif !defined (__APPLE__)

FileSpec
Symbols::LocateExecutableObjectFile (const ModuleSpec &module_spec)
{
    // FIXME
    return FileSpec();
}

FileSpec
Symbols::LocateExecutableSymbolFile (const ModuleSpec &module_spec)
{
    // FIXME
    return FileSpec();
}

FileSpec
Symbols::FindSymbolFileInBundle (const FileSpec& symfile_bundle,
                                 const lldb_private::UUID *uuid,
                                 const ArchSpec *arch)
{
    return FileSpec();
}

bool
Symbols::DownloadObjectAndSymbolFile (ModuleSpec &module_spec, bool force_lookup)
{
    // Fill in the module_spec.GetFileSpec() for the object file and/or the
    // module_spec.GetSymbolFileSpec() for the debug symbols file.
    return false;
}

#endif
