//===--------------------- ModuleCache.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ModuleCache.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/LockFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"

#include <assert.h>

#include <cstdio>

using namespace lldb;
using namespace lldb_private;

namespace {

const char* kModulesSubdir = ".cache";
const char* kLockFileName = ".lock";
const char* kTempFileName = ".temp";

FileSpec
JoinPath (const FileSpec &path1, const char* path2)
{
    FileSpec result_spec (path1);
    result_spec.AppendPathComponent (path2);
    return result_spec;
}

Error
MakeDirectory (const FileSpec &dir_path)
{
    if (dir_path.Exists ())
    {
        if (!dir_path.IsDirectory ())
            return Error ("Invalid existing path");

        return Error ();
    }

    return FileSystem::MakeDirectory(dir_path, eFilePermissionsDirectoryDefault);
}

FileSpec
GetModuleDirectory (const FileSpec &root_dir_spec, const UUID &uuid)
{
    const auto modules_dir_spec = JoinPath (root_dir_spec, kModulesSubdir);
    return JoinPath (modules_dir_spec, uuid.GetAsString ().c_str ());
}

Error
CreateHostSysRootModuleLink (const FileSpec &root_dir_spec, const char *hostname, const FileSpec &platform_module_spec, const FileSpec &local_module_spec)
{
    const auto sysroot_module_path_spec = JoinPath (
        JoinPath (root_dir_spec, hostname), platform_module_spec.GetPath ().c_str ());
    if (sysroot_module_path_spec.Exists())
        return Error ();

    const auto error = MakeDirectory (FileSpec (sysroot_module_path_spec.GetDirectory ().AsCString (), false));
    if (error.Fail ())
        return error;

    return FileSystem::Hardlink(sysroot_module_path_spec, local_module_spec);
}

}  // namespace

Error
ModuleCache::Put (const FileSpec &root_dir_spec,
                  const char *hostname,
                  const ModuleSpec &module_spec,
                  const FileSpec &tmp_file)
{
    const auto module_spec_dir = GetModuleDirectory (root_dir_spec, module_spec.GetUUID ());
    const auto module_file_path = JoinPath (module_spec_dir, module_spec.GetFileSpec ().GetFilename ().AsCString ());

    const auto tmp_file_path = tmp_file.GetPath ();
    const auto err_code = llvm::sys::fs::rename (tmp_file_path.c_str (), module_file_path.GetPath ().c_str ());
    if (err_code)
        return Error ("Failed to rename file %s to %s: %s",
                      tmp_file_path.c_str (), module_file_path.GetPath ().c_str (), err_code.message ().c_str ());

    const auto error = CreateHostSysRootModuleLink(root_dir_spec, hostname, module_spec.GetFileSpec(), module_file_path);
    if (error.Fail ())
        return Error ("Failed to create link to %s: %s", module_file_path.GetPath ().c_str (), error.AsCString ());
    return Error ();
}

Error
ModuleCache::Get (const FileSpec &root_dir_spec,
                  const char *hostname,
                  const ModuleSpec &module_spec,
                  ModuleSP &cached_module_sp,
                  bool *did_create_ptr)
{
    const auto find_it = m_loaded_modules.find (module_spec.GetUUID ().GetAsString());
    if (find_it != m_loaded_modules.end ())
    {
        cached_module_sp = (*find_it).second.lock ();
        if (cached_module_sp)
            return Error ();
        m_loaded_modules.erase (find_it);
    }

    const auto module_spec_dir = GetModuleDirectory (root_dir_spec, module_spec.GetUUID ());
    const auto module_file_path = JoinPath (module_spec_dir, module_spec.GetFileSpec ().GetFilename ().AsCString ());

    if (!module_file_path.Exists ())
        return Error ("Module %s not found", module_file_path.GetPath ().c_str ());
    if (module_file_path.GetByteSize () != module_spec.GetObjectSize ())
        return Error ("Module %s has invalid file size", module_file_path.GetPath ().c_str ());

    // We may have already cached module but downloaded from an another host - in this case let's create a link to it.
    const auto error = CreateHostSysRootModuleLink(root_dir_spec, hostname, module_spec.GetFileSpec(), module_file_path);
    if (error.Fail ())
        return Error ("Failed to create link to %s: %s", module_file_path.GetPath().c_str(), error.AsCString());

    auto cached_module_spec (module_spec);
    cached_module_spec.GetUUID ().Clear ();  // Clear UUID since it may contain md5 content hash instead of real UUID.
    cached_module_spec.GetFileSpec () = module_file_path;
    cached_module_spec.GetPlatformFileSpec () = module_spec.GetFileSpec ();
    cached_module_sp.reset (new Module (cached_module_spec));
    if (did_create_ptr)
        *did_create_ptr = true;

    m_loaded_modules.insert (std::make_pair (module_spec.GetUUID ().GetAsString (), cached_module_sp));

    return Error ();
}

Error
ModuleCache::GetAndPut (const FileSpec &root_dir_spec,
                        const char *hostname,
                        const ModuleSpec &module_spec,
                        const Downloader &downloader,
                        lldb::ModuleSP &cached_module_sp,
                        bool *did_create_ptr)
{
    const auto module_spec_dir = GetModuleDirectory (root_dir_spec, module_spec.GetUUID ());
    auto error = MakeDirectory (module_spec_dir);
    if (error.Fail ())
        return error;

    // Open lock file.
    const auto lock_file_spec = JoinPath (module_spec_dir, kLockFileName);
    File lock_file (lock_file_spec, File::eOpenOptionWrite | File::eOpenOptionCanCreate | File::eOpenOptionCloseOnExec);
    if (!lock_file)
    {
        error.SetErrorToErrno ();
        return Error("Failed to open lock file %s: %s", lock_file_spec.GetPath ().c_str (), error.AsCString ());
    }
    LockFile lock (lock_file.GetDescriptor ());
    error = lock.WriteLock (0, 1);
    if (error.Fail ())
        return Error("Failed to lock file %s:%s", lock_file_spec.GetPath ().c_str (), error.AsCString ());

    // Check local cache for a module.
    error = Get (root_dir_spec, hostname, module_spec, cached_module_sp, did_create_ptr);
    if (error.Success ())
        return error;

    const auto tmp_download_file_spec = JoinPath (module_spec_dir, kTempFileName);
    error = downloader (module_spec, tmp_download_file_spec);
    llvm::FileRemover tmp_file_remover (tmp_download_file_spec.GetPath ().c_str ());
    if (error.Fail ())
        return Error("Failed to download module: %s", error.AsCString ());

    // Put downloaded file into local module cache.
    error = Put (root_dir_spec, hostname, module_spec, tmp_download_file_spec);
    if (error.Fail ())
        return Error ("Failed to put module into cache: %s", error.AsCString ());

    tmp_file_remover.releaseFile ();
    return Get (root_dir_spec, hostname, module_spec, cached_module_sp, did_create_ptr);
}
