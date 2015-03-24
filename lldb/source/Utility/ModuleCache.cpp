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
#include "lldb/Host/FileSystem.h"
#include "llvm/Support/FileSystem.h"

#include <assert.h>

#include <cstdio>
#
using namespace lldb;
using namespace lldb_private;

namespace {

const char* kModulesSubdir = ".cache";

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

    return FileSystem::MakeDirectory (dir_path.GetPath ().c_str (),
                                      eFilePermissionsDirectoryDefault);
}

}  // namespace

Error
ModuleCache::Put (const FileSpec &root_dir_spec,
                  const char *hostname,
                  const ModuleSpec &module_spec,
                  const FileSpec &tmp_file)
{
    const auto module_spec_dir = GetModuleDirectory (root_dir_spec, module_spec.GetUUID ());
    auto error = MakeDirectory (module_spec_dir);
    if (error.Fail ())
        return error;

    const auto module_file_path = JoinPath (module_spec_dir, module_spec.GetFileSpec ().GetFilename ().AsCString ());

    const auto tmp_file_path = tmp_file.GetPath ();
    const auto err_code = llvm::sys::fs::copy_file (tmp_file_path.c_str (), module_file_path.GetPath ().c_str ());
    if (err_code)
    {
        error.SetErrorStringWithFormat ("failed to copy file %s to %s: %s",
                                        tmp_file_path.c_str (),
                                        module_file_path.GetPath ().c_str (),
                                        err_code.message ().c_str ());
    }

    // Create sysroot link to a module.
    const auto sysroot_module_path_spec = GetHostSysRootModulePath (root_dir_spec, hostname, module_spec.GetFileSpec ());
    return CreateHostSysRootModuleSymLink (sysroot_module_path_spec, module_file_path);
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

    const auto module_spec_dir = GetModuleDirectory (root_dir_spec,  module_spec.GetUUID ());
    const auto module_file_path = JoinPath (module_spec_dir, module_spec.GetFileSpec ().GetFilename ().AsCString ());

    if (!module_file_path.Exists ())
        return Error ("module %s not found", module_file_path.GetPath ().c_str ());

    // We may have already cached module but downloaded from an another host - in this case let's create a symlink to it.
    const auto sysroot_module_path_spec = GetHostSysRootModulePath (root_dir_spec, hostname, module_spec.GetFileSpec ());
    if (!sysroot_module_path_spec.Exists ())
        CreateHostSysRootModuleSymLink (sysroot_module_path_spec, module_spec.GetFileSpec ());

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

FileSpec
ModuleCache::GetModuleDirectory (const FileSpec &root_dir_spec, const UUID &uuid)
{
    const auto modules_dir_spec = JoinPath (root_dir_spec, kModulesSubdir);
    return JoinPath (modules_dir_spec, uuid.GetAsString ().c_str ());
}

FileSpec
ModuleCache::GetHostSysRootModulePath (const FileSpec &root_dir_spec, const char *hostname, const FileSpec &platform_module_spec)
{
    const auto sysroot_dir = JoinPath (root_dir_spec, hostname);
    return JoinPath (sysroot_dir, platform_module_spec.GetPath ().c_str ());
}

Error
ModuleCache::CreateHostSysRootModuleSymLink (const FileSpec &sysroot_module_path_spec, const FileSpec &module_file_path)
{
    const auto error = MakeDirectory (FileSpec (sysroot_module_path_spec.GetDirectory ().AsCString (), false));
    if (error.Fail ())
        return error;

    return FileSystem::Symlink (sysroot_module_path_spec.GetPath ().c_str (),
                                module_file_path.GetPath ().c_str ());
}
