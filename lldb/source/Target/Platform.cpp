//===-- Platform.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <csignal>
#include <fstream>
#include <memory>
#include <vector>

#include "lldb/Breakpoint/BreakpointIDList.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/OptionValueFileSpec.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Interpreter/Property.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/ModuleCache.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StructuredData.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

// Define these constants from POSIX mman.h rather than include the file so
// that they will be correct even when compiled on Linux.
#define MAP_PRIVATE 2
#define MAP_ANON 0x1000

using namespace lldb;
using namespace lldb_private;

static uint32_t g_initialize_count = 0;

// Use a singleton function for g_local_platform_sp to avoid init constructors
// since LLDB is often part of a shared library
static PlatformSP &GetHostPlatformSP() {
  static PlatformSP g_platform_sp;
  return g_platform_sp;
}

const char *Platform::GetHostPlatformName() { return "host"; }

namespace {

#define LLDB_PROPERTIES_platform
#include "TargetProperties.inc"

enum {
#define LLDB_PROPERTIES_platform
#include "TargetPropertiesEnum.inc"
};

} // namespace

ConstString PlatformProperties::GetSettingName() {
  static ConstString g_setting_name("platform");
  return g_setting_name;
}

PlatformProperties::PlatformProperties() {
  m_collection_sp = std::make_shared<OptionValueProperties>(GetSettingName());
  m_collection_sp->Initialize(g_platform_properties);

  auto module_cache_dir = GetModuleCacheDirectory();
  if (module_cache_dir)
    return;

  llvm::SmallString<64> user_home_dir;
  if (!FileSystem::Instance().GetHomeDirectory(user_home_dir))
    return;

  module_cache_dir = FileSpec(user_home_dir.c_str());
  module_cache_dir.AppendPathComponent(".lldb");
  module_cache_dir.AppendPathComponent("module_cache");
  SetDefaultModuleCacheDirectory(module_cache_dir);
  SetModuleCacheDirectory(module_cache_dir);
}

bool PlatformProperties::GetUseModuleCache() const {
  const auto idx = ePropertyUseModuleCache;
  return m_collection_sp->GetPropertyAtIndexAsBoolean(
      nullptr, idx, g_platform_properties[idx].default_uint_value != 0);
}

bool PlatformProperties::SetUseModuleCache(bool use_module_cache) {
  return m_collection_sp->SetPropertyAtIndexAsBoolean(
      nullptr, ePropertyUseModuleCache, use_module_cache);
}

FileSpec PlatformProperties::GetModuleCacheDirectory() const {
  return m_collection_sp->GetPropertyAtIndexAsFileSpec(
      nullptr, ePropertyModuleCacheDirectory);
}

bool PlatformProperties::SetModuleCacheDirectory(const FileSpec &dir_spec) {
  return m_collection_sp->SetPropertyAtIndexAsFileSpec(
      nullptr, ePropertyModuleCacheDirectory, dir_spec);
}

void PlatformProperties::SetDefaultModuleCacheDirectory(
    const FileSpec &dir_spec) {
  auto f_spec_opt = m_collection_sp->GetPropertyAtIndexAsOptionValueFileSpec(
        nullptr, false, ePropertyModuleCacheDirectory);
  assert(f_spec_opt);
  f_spec_opt->SetDefaultValue(dir_spec);
}

/// Get the native host platform plug-in.
///
/// There should only be one of these for each host that LLDB runs
/// upon that should be statically compiled in and registered using
/// preprocessor macros or other similar build mechanisms.
///
/// This platform will be used as the default platform when launching
/// or attaching to processes unless another platform is specified.
PlatformSP Platform::GetHostPlatform() { return GetHostPlatformSP(); }

static std::vector<PlatformSP> &GetPlatformList() {
  static std::vector<PlatformSP> g_platform_list;
  return g_platform_list;
}

static std::recursive_mutex &GetPlatformListMutex() {
  static std::recursive_mutex g_mutex;
  return g_mutex;
}

void Platform::Initialize() { g_initialize_count++; }

void Platform::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      std::lock_guard<std::recursive_mutex> guard(GetPlatformListMutex());
      GetPlatformList().clear();
    }
  }
}

PlatformProperties &Platform::GetGlobalPlatformProperties() {
  static PlatformProperties g_settings;
  return g_settings;
}

void Platform::SetHostPlatform(const lldb::PlatformSP &platform_sp) {
  // The native platform should use its static void Platform::Initialize()
  // function to register itself as the native platform.
  GetHostPlatformSP() = platform_sp;

  if (platform_sp) {
    std::lock_guard<std::recursive_mutex> guard(GetPlatformListMutex());
    GetPlatformList().push_back(platform_sp);
  }
}

Status Platform::GetFileWithUUID(const FileSpec &platform_file,
                                 const UUID *uuid_ptr, FileSpec &local_file) {
  // Default to the local case
  local_file = platform_file;
  return Status();
}

FileSpecList
Platform::LocateExecutableScriptingResources(Target *target, Module &module,
                                             Stream *feedback_stream) {
  return FileSpecList();
}

// PlatformSP
// Platform::FindPlugin (Process *process, ConstString plugin_name)
//{
//    PlatformCreateInstance create_callback = nullptr;
//    if (plugin_name)
//    {
//        create_callback  =
//        PluginManager::GetPlatformCreateCallbackForPluginName (plugin_name);
//        if (create_callback)
//        {
//            ArchSpec arch;
//            if (process)
//            {
//                arch = process->GetTarget().GetArchitecture();
//            }
//            PlatformSP platform_sp(create_callback(process, &arch));
//            if (platform_sp)
//                return platform_sp;
//        }
//    }
//    else
//    {
//        for (uint32_t idx = 0; (create_callback =
//        PluginManager::GetPlatformCreateCallbackAtIndex(idx)) != nullptr;
//        ++idx)
//        {
//            PlatformSP platform_sp(create_callback(process, nullptr));
//            if (platform_sp)
//                return platform_sp;
//        }
//    }
//    return PlatformSP();
//}

Status Platform::GetSharedModule(
    const ModuleSpec &module_spec, Process *process, ModuleSP &module_sp,
    const FileSpecList *module_search_paths_ptr,
    llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules, bool *did_create_ptr) {
  if (IsHost())
    return ModuleList::GetSharedModule(module_spec, module_sp,
                                       module_search_paths_ptr, old_modules,
                                       did_create_ptr, false);

  // Module resolver lambda.
  auto resolver = [&](const ModuleSpec &spec) {
    Status error(eErrorTypeGeneric);
    ModuleSpec resolved_spec;
    // Check if we have sysroot set.
    if (m_sdk_sysroot) {
      // Prepend sysroot to module spec.
      resolved_spec = spec;
      resolved_spec.GetFileSpec().PrependPathComponent(
          m_sdk_sysroot.GetStringRef());
      // Try to get shared module with resolved spec.
      error = ModuleList::GetSharedModule(resolved_spec, module_sp,
                                          module_search_paths_ptr, old_modules,
                                          did_create_ptr, false);
    }
    // If we don't have sysroot or it didn't work then
    // try original module spec.
    if (!error.Success()) {
      resolved_spec = spec;
      error = ModuleList::GetSharedModule(resolved_spec, module_sp,
                                          module_search_paths_ptr, old_modules,
                                          did_create_ptr, false);
    }
    if (error.Success() && module_sp)
      module_sp->SetPlatformFileSpec(resolved_spec.GetFileSpec());
    return error;
  };

  return GetRemoteSharedModule(module_spec, process, module_sp, resolver,
                               did_create_ptr);
}

bool Platform::GetModuleSpec(const FileSpec &module_file_spec,
                             const ArchSpec &arch, ModuleSpec &module_spec) {
  ModuleSpecList module_specs;
  if (ObjectFile::GetModuleSpecifications(module_file_spec, 0, 0,
                                          module_specs) == 0)
    return false;

  ModuleSpec matched_module_spec;
  return module_specs.FindMatchingModuleSpec(ModuleSpec(module_file_spec, arch),
                                             module_spec);
}

PlatformSP Platform::Find(ConstString name) {
  if (name) {
    static ConstString g_host_platform_name("host");
    if (name == g_host_platform_name)
      return GetHostPlatform();

    std::lock_guard<std::recursive_mutex> guard(GetPlatformListMutex());
    for (const auto &platform_sp : GetPlatformList()) {
      if (platform_sp->GetName() == name)
        return platform_sp;
    }
  }
  return PlatformSP();
}

PlatformSP Platform::Create(ConstString name, Status &error) {
  PlatformCreateInstance create_callback = nullptr;
  lldb::PlatformSP platform_sp;
  if (name) {
    static ConstString g_host_platform_name("host");
    if (name == g_host_platform_name)
      return GetHostPlatform();

    create_callback = PluginManager::GetPlatformCreateCallbackForPluginName(
        name.GetStringRef());
    if (create_callback)
      platform_sp = create_callback(true, nullptr);
    else
      error.SetErrorStringWithFormat(
          "unable to find a plug-in for the platform named \"%s\"",
          name.GetCString());
  } else
    error.SetErrorString("invalid platform name");

  if (platform_sp) {
    std::lock_guard<std::recursive_mutex> guard(GetPlatformListMutex());
    GetPlatformList().push_back(platform_sp);
  }

  return platform_sp;
}

PlatformSP Platform::Create(const ArchSpec &arch, ArchSpec *platform_arch_ptr,
                            Status &error) {
  lldb::PlatformSP platform_sp;
  if (arch.IsValid()) {
    // Scope for locker
    {
      // First try exact arch matches across all platforms already created
      std::lock_guard<std::recursive_mutex> guard(GetPlatformListMutex());
      for (const auto &platform_sp : GetPlatformList()) {
        if (platform_sp->IsCompatibleArchitecture(arch, true,
                                                  platform_arch_ptr))
          return platform_sp;
      }

      // Next try compatible arch matches across all platforms already created
      for (const auto &platform_sp : GetPlatformList()) {
        if (platform_sp->IsCompatibleArchitecture(arch, false,
                                                  platform_arch_ptr))
          return platform_sp;
      }
    }

    PlatformCreateInstance create_callback;
    // First try exact arch matches across all platform plug-ins
    uint32_t idx;
    for (idx = 0; (create_callback =
                       PluginManager::GetPlatformCreateCallbackAtIndex(idx));
         ++idx) {
      if (create_callback) {
        platform_sp = create_callback(false, &arch);
        if (platform_sp &&
            platform_sp->IsCompatibleArchitecture(arch, true,
                                                  platform_arch_ptr)) {
          std::lock_guard<std::recursive_mutex> guard(GetPlatformListMutex());
          GetPlatformList().push_back(platform_sp);
          return platform_sp;
        }
      }
    }
    // Next try compatible arch matches across all platform plug-ins
    for (idx = 0; (create_callback =
                       PluginManager::GetPlatformCreateCallbackAtIndex(idx));
         ++idx) {
      if (create_callback) {
        platform_sp = create_callback(false, &arch);
        if (platform_sp &&
            platform_sp->IsCompatibleArchitecture(arch, false,
                                                  platform_arch_ptr)) {
          std::lock_guard<std::recursive_mutex> guard(GetPlatformListMutex());
          GetPlatformList().push_back(platform_sp);
          return platform_sp;
        }
      }
    }
  } else
    error.SetErrorString("invalid platform name");
  if (platform_arch_ptr)
    platform_arch_ptr->Clear();
  platform_sp.reset();
  return platform_sp;
}

ArchSpec Platform::GetAugmentedArchSpec(Platform *platform, llvm::StringRef triple) {
  if (platform)
    return platform->GetAugmentedArchSpec(triple);
  return HostInfo::GetAugmentedArchSpec(triple);
}

/// Default Constructor
Platform::Platform(bool is_host)
    : m_is_host(is_host), m_os_version_set_while_connected(false),
      m_system_arch_set_while_connected(false), m_sdk_sysroot(), m_sdk_build(),
      m_working_dir(), m_remote_url(), m_name(), m_system_arch(), m_mutex(),
      m_max_uid_name_len(0), m_max_gid_name_len(0), m_supports_rsync(false),
      m_rsync_opts(), m_rsync_prefix(), m_supports_ssh(false), m_ssh_opts(),
      m_ignores_remote_hostname(false), m_trap_handlers(),
      m_calculated_trap_handlers(false),
      m_module_cache(std::make_unique<ModuleCache>()) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_OBJECT));
  LLDB_LOGF(log, "%p Platform::Platform()", static_cast<void *>(this));
}

Platform::~Platform() = default;

void Platform::GetStatus(Stream &strm) {
  strm.Format("  Platform: {0}\n", GetPluginName());

  ArchSpec arch(GetSystemArchitecture());
  if (arch.IsValid()) {
    if (!arch.GetTriple().str().empty()) {
      strm.Printf("    Triple: ");
      arch.DumpTriple(strm.AsRawOstream());
      strm.EOL();
    }
  }

  llvm::VersionTuple os_version = GetOSVersion();
  if (!os_version.empty()) {
    strm.Format("OS Version: {0}", os_version.getAsString());

    if (llvm::Optional<std::string> s = GetOSBuildString())
      strm.Format(" ({0})", *s);

    strm.EOL();
  }

  if (IsHost()) {
    strm.Printf("  Hostname: %s\n", GetHostname());
  } else {
    const bool is_connected = IsConnected();
    if (is_connected)
      strm.Printf("  Hostname: %s\n", GetHostname());
    strm.Printf(" Connected: %s\n", is_connected ? "yes" : "no");
  }

  if (GetSDKRootDirectory()) {
    strm.Format("   Sysroot: {0}\n", GetSDKRootDirectory());
  }
  if (GetWorkingDirectory()) {
    strm.Printf("WorkingDir: %s\n", GetWorkingDirectory().GetCString());
  }
  if (!IsConnected())
    return;

  std::string specific_info(GetPlatformSpecificConnectionInformation());

  if (!specific_info.empty())
    strm.Printf("Platform-specific connection: %s\n", specific_info.c_str());

  if (llvm::Optional<std::string> s = GetOSKernelDescription())
    strm.Format("    Kernel: {0}\n", *s);
}

llvm::VersionTuple Platform::GetOSVersion(Process *process) {
  std::lock_guard<std::mutex> guard(m_mutex);

  if (IsHost()) {
    if (m_os_version.empty()) {
      // We have a local host platform
      m_os_version = HostInfo::GetOSVersion();
      m_os_version_set_while_connected = !m_os_version.empty();
    }
  } else {
    // We have a remote platform. We can only fetch the remote
    // OS version if we are connected, and we don't want to do it
    // more than once.

    const bool is_connected = IsConnected();

    bool fetch = false;
    if (!m_os_version.empty()) {
      // We have valid OS version info, check to make sure it wasn't manually
      // set prior to connecting. If it was manually set prior to connecting,
      // then lets fetch the actual OS version info if we are now connected.
      if (is_connected && !m_os_version_set_while_connected)
        fetch = true;
    } else {
      // We don't have valid OS version info, fetch it if we are connected
      fetch = is_connected;
    }

    if (fetch)
      m_os_version_set_while_connected = GetRemoteOSVersion();
  }

  if (!m_os_version.empty())
    return m_os_version;
  if (process) {
    // Check with the process in case it can answer the question if a process
    // was provided
    return process->GetHostOSVersion();
  }
  return llvm::VersionTuple();
}

llvm::Optional<std::string> Platform::GetOSBuildString() {
  if (IsHost())
    return HostInfo::GetOSBuildString();
  return GetRemoteOSBuildString();
}

llvm::Optional<std::string> Platform::GetOSKernelDescription() {
  if (IsHost())
    return HostInfo::GetOSKernelDescription();
  return GetRemoteOSKernelDescription();
}

void Platform::AddClangModuleCompilationOptions(
    Target *target, std::vector<std::string> &options) {
  std::vector<std::string> default_compilation_options = {
      "-x", "c++", "-Xclang", "-nostdsysteminc", "-Xclang", "-nostdsysteminc"};

  options.insert(options.end(), default_compilation_options.begin(),
                 default_compilation_options.end());
}

FileSpec Platform::GetWorkingDirectory() {
  if (IsHost()) {
    llvm::SmallString<64> cwd;
    if (llvm::sys::fs::current_path(cwd))
      return {};
    else {
      FileSpec file_spec(cwd);
      FileSystem::Instance().Resolve(file_spec);
      return file_spec;
    }
  } else {
    if (!m_working_dir)
      m_working_dir = GetRemoteWorkingDirectory();
    return m_working_dir;
  }
}

struct RecurseCopyBaton {
  const FileSpec &dst;
  Platform *platform_ptr;
  Status error;
};

static FileSystem::EnumerateDirectoryResult
RecurseCopy_Callback(void *baton, llvm::sys::fs::file_type ft,
                     llvm::StringRef path) {
  RecurseCopyBaton *rc_baton = (RecurseCopyBaton *)baton;
  FileSpec src(path);
  namespace fs = llvm::sys::fs;
  switch (ft) {
  case fs::file_type::fifo_file:
  case fs::file_type::socket_file:
    // we have no way to copy pipes and sockets - ignore them and continue
    return FileSystem::eEnumerateDirectoryResultNext;
    break;

  case fs::file_type::directory_file: {
    // make the new directory and get in there
    FileSpec dst_dir = rc_baton->dst;
    if (!dst_dir.GetFilename())
      dst_dir.GetFilename() = src.GetLastPathComponent();
    Status error = rc_baton->platform_ptr->MakeDirectory(
        dst_dir, lldb::eFilePermissionsDirectoryDefault);
    if (error.Fail()) {
      rc_baton->error.SetErrorStringWithFormat(
          "unable to setup directory %s on remote end", dst_dir.GetCString());
      return FileSystem::eEnumerateDirectoryResultQuit; // got an error, bail out
    }

    // now recurse
    std::string src_dir_path(src.GetPath());

    // Make a filespec that only fills in the directory of a FileSpec so when
    // we enumerate we can quickly fill in the filename for dst copies
    FileSpec recurse_dst;
    recurse_dst.GetDirectory().SetCString(dst_dir.GetPath().c_str());
    RecurseCopyBaton rc_baton2 = {recurse_dst, rc_baton->platform_ptr,
                                  Status()};
    FileSystem::Instance().EnumerateDirectory(src_dir_path, true, true, true,
                                              RecurseCopy_Callback, &rc_baton2);
    if (rc_baton2.error.Fail()) {
      rc_baton->error.SetErrorString(rc_baton2.error.AsCString());
      return FileSystem::eEnumerateDirectoryResultQuit; // got an error, bail out
    }
    return FileSystem::eEnumerateDirectoryResultNext;
  } break;

  case fs::file_type::symlink_file: {
    // copy the file and keep going
    FileSpec dst_file = rc_baton->dst;
    if (!dst_file.GetFilename())
      dst_file.GetFilename() = src.GetFilename();

    FileSpec src_resolved;

    rc_baton->error = FileSystem::Instance().Readlink(src, src_resolved);

    if (rc_baton->error.Fail())
      return FileSystem::eEnumerateDirectoryResultQuit; // got an error, bail out

    rc_baton->error =
        rc_baton->platform_ptr->CreateSymlink(dst_file, src_resolved);

    if (rc_baton->error.Fail())
      return FileSystem::eEnumerateDirectoryResultQuit; // got an error, bail out

    return FileSystem::eEnumerateDirectoryResultNext;
  } break;

  case fs::file_type::regular_file: {
    // copy the file and keep going
    FileSpec dst_file = rc_baton->dst;
    if (!dst_file.GetFilename())
      dst_file.GetFilename() = src.GetFilename();
    Status err = rc_baton->platform_ptr->PutFile(src, dst_file);
    if (err.Fail()) {
      rc_baton->error.SetErrorString(err.AsCString());
      return FileSystem::eEnumerateDirectoryResultQuit; // got an error, bail out
    }
    return FileSystem::eEnumerateDirectoryResultNext;
  } break;

  default:
    rc_baton->error.SetErrorStringWithFormat(
        "invalid file detected during copy: %s", src.GetPath().c_str());
    return FileSystem::eEnumerateDirectoryResultQuit; // got an error, bail out
    break;
  }
  llvm_unreachable("Unhandled file_type!");
}

Status Platform::Install(const FileSpec &src, const FileSpec &dst) {
  Status error;

  Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
  LLDB_LOGF(log, "Platform::Install (src='%s', dst='%s')",
            src.GetPath().c_str(), dst.GetPath().c_str());
  FileSpec fixed_dst(dst);

  if (!fixed_dst.GetFilename())
    fixed_dst.GetFilename() = src.GetFilename();

  FileSpec working_dir = GetWorkingDirectory();

  if (dst) {
    if (dst.GetDirectory()) {
      const char first_dst_dir_char = dst.GetDirectory().GetCString()[0];
      if (first_dst_dir_char == '/' || first_dst_dir_char == '\\') {
        fixed_dst.GetDirectory() = dst.GetDirectory();
      }
      // If the fixed destination file doesn't have a directory yet, then we
      // must have a relative path. We will resolve this relative path against
      // the platform's working directory
      if (!fixed_dst.GetDirectory()) {
        FileSpec relative_spec;
        std::string path;
        if (working_dir) {
          relative_spec = working_dir;
          relative_spec.AppendPathComponent(dst.GetPath());
          fixed_dst.GetDirectory() = relative_spec.GetDirectory();
        } else {
          error.SetErrorStringWithFormat(
              "platform working directory must be valid for relative path '%s'",
              dst.GetPath().c_str());
          return error;
        }
      }
    } else {
      if (working_dir) {
        fixed_dst.GetDirectory().SetCString(working_dir.GetCString());
      } else {
        error.SetErrorStringWithFormat(
            "platform working directory must be valid for relative path '%s'",
            dst.GetPath().c_str());
        return error;
      }
    }
  } else {
    if (working_dir) {
      fixed_dst.GetDirectory().SetCString(working_dir.GetCString());
    } else {
      error.SetErrorStringWithFormat("platform working directory must be valid "
                                     "when destination directory is empty");
      return error;
    }
  }

  LLDB_LOGF(log, "Platform::Install (src='%s', dst='%s') fixed_dst='%s'",
            src.GetPath().c_str(), dst.GetPath().c_str(),
            fixed_dst.GetPath().c_str());

  if (GetSupportsRSync()) {
    error = PutFile(src, dst);
  } else {
    namespace fs = llvm::sys::fs;
    switch (fs::get_file_type(src.GetPath(), false)) {
    case fs::file_type::directory_file: {
      llvm::sys::fs::remove(fixed_dst.GetPath());
      uint32_t permissions = FileSystem::Instance().GetPermissions(src);
      if (permissions == 0)
        permissions = eFilePermissionsDirectoryDefault;
      error = MakeDirectory(fixed_dst, permissions);
      if (error.Success()) {
        // Make a filespec that only fills in the directory of a FileSpec so
        // when we enumerate we can quickly fill in the filename for dst copies
        FileSpec recurse_dst;
        recurse_dst.GetDirectory().SetCString(fixed_dst.GetCString());
        std::string src_dir_path(src.GetPath());
        RecurseCopyBaton baton = {recurse_dst, this, Status()};
        FileSystem::Instance().EnumerateDirectory(
            src_dir_path, true, true, true, RecurseCopy_Callback, &baton);
        return baton.error;
      }
    } break;

    case fs::file_type::regular_file:
      llvm::sys::fs::remove(fixed_dst.GetPath());
      error = PutFile(src, fixed_dst);
      break;

    case fs::file_type::symlink_file: {
      llvm::sys::fs::remove(fixed_dst.GetPath());
      FileSpec src_resolved;
      error = FileSystem::Instance().Readlink(src, src_resolved);
      if (error.Success())
        error = CreateSymlink(dst, src_resolved);
    } break;
    case fs::file_type::fifo_file:
      error.SetErrorString("platform install doesn't handle pipes");
      break;
    case fs::file_type::socket_file:
      error.SetErrorString("platform install doesn't handle sockets");
      break;
    default:
      error.SetErrorString(
          "platform install doesn't handle non file or directory items");
      break;
    }
  }
  return error;
}

bool Platform::SetWorkingDirectory(const FileSpec &file_spec) {
  if (IsHost()) {
    Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
    LLDB_LOG(log, "{0}", file_spec);
    if (std::error_code ec = llvm::sys::fs::set_current_path(file_spec.GetPath())) {
      LLDB_LOG(log, "error: {0}", ec.message());
      return false;
    }
    return true;
  } else {
    m_working_dir.Clear();
    return SetRemoteWorkingDirectory(file_spec);
  }
}

Status Platform::MakeDirectory(const FileSpec &file_spec,
                               uint32_t permissions) {
  if (IsHost())
    return llvm::sys::fs::create_directory(file_spec.GetPath(), permissions);
  else {
    Status error;
    error.SetErrorStringWithFormatv("remote platform {0} doesn't support {1}",
                                    GetPluginName(), LLVM_PRETTY_FUNCTION);
    return error;
  }
}

Status Platform::GetFilePermissions(const FileSpec &file_spec,
                                    uint32_t &file_permissions) {
  if (IsHost()) {
    auto Value = llvm::sys::fs::getPermissions(file_spec.GetPath());
    if (Value)
      file_permissions = Value.get();
    return Status(Value.getError());
  } else {
    Status error;
    error.SetErrorStringWithFormatv("remote platform {0} doesn't support {1}",
                                    GetPluginName(), LLVM_PRETTY_FUNCTION);
    return error;
  }
}

Status Platform::SetFilePermissions(const FileSpec &file_spec,
                                    uint32_t file_permissions) {
  if (IsHost()) {
    auto Perms = static_cast<llvm::sys::fs::perms>(file_permissions);
    return llvm::sys::fs::setPermissions(file_spec.GetPath(), Perms);
  } else {
    Status error;
    error.SetErrorStringWithFormatv("remote platform {0} doesn't support {1}",
                                    GetPluginName(), LLVM_PRETTY_FUNCTION);
    return error;
  }
}

ConstString Platform::GetName() { return ConstString(GetPluginName()); }

const char *Platform::GetHostname() {
  if (IsHost())
    return "127.0.0.1";

  if (m_name.empty())
    return nullptr;
  return m_name.c_str();
}

ConstString Platform::GetFullNameForDylib(ConstString basename) {
  return basename;
}

bool Platform::SetRemoteWorkingDirectory(const FileSpec &working_dir) {
  Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
  LLDB_LOGF(log, "Platform::SetRemoteWorkingDirectory('%s')",
            working_dir.GetCString());
  m_working_dir = working_dir;
  return true;
}

bool Platform::SetOSVersion(llvm::VersionTuple version) {
  if (IsHost()) {
    // We don't need anyone setting the OS version for the host platform, we
    // should be able to figure it out by calling HostInfo::GetOSVersion(...).
    return false;
  } else {
    // We have a remote platform, allow setting the target OS version if we
    // aren't connected, since if we are connected, we should be able to
    // request the remote OS version from the connected platform.
    if (IsConnected())
      return false;
    else {
      // We aren't connected and we might want to set the OS version ahead of
      // time before we connect so we can peruse files and use a local SDK or
      // PDK cache of support files to disassemble or do other things.
      m_os_version = version;
      return true;
    }
  }
  return false;
}

Status
Platform::ResolveExecutable(const ModuleSpec &module_spec,
                            lldb::ModuleSP &exe_module_sp,
                            const FileSpecList *module_search_paths_ptr) {
  Status error;

  if (FileSystem::Instance().Exists(module_spec.GetFileSpec())) {
    if (module_spec.GetArchitecture().IsValid()) {
      error = ModuleList::GetSharedModule(module_spec, exe_module_sp,
                                          module_search_paths_ptr, nullptr,
                                          nullptr);
    } else {
      // No valid architecture was specified, ask the platform for the
      // architectures that we should be using (in the correct order) and see
      // if we can find a match that way
      ModuleSpec arch_module_spec(module_spec);
      for (const ArchSpec &arch : GetSupportedArchitectures()) {
        arch_module_spec.GetArchitecture() = arch;
        error = ModuleList::GetSharedModule(arch_module_spec, exe_module_sp,
                                            module_search_paths_ptr, nullptr,
                                            nullptr);
        // Did we find an executable using one of the
        if (error.Success() && exe_module_sp)
          break;
      }
    }
  } else {
    error.SetErrorStringWithFormat(
        "'%s' does not exist", module_spec.GetFileSpec().GetPath().c_str());
  }
  return error;
}

Status
Platform::ResolveRemoteExecutable(const ModuleSpec &module_spec,
                            lldb::ModuleSP &exe_module_sp,
                            const FileSpecList *module_search_paths_ptr) {
  Status error;

  // We may connect to a process and use the provided executable (Don't use
  // local $PATH).
  ModuleSpec resolved_module_spec(module_spec);

  // Resolve any executable within a bundle on MacOSX
  Host::ResolveExecutableInBundle(resolved_module_spec.GetFileSpec());

  if (FileSystem::Instance().Exists(resolved_module_spec.GetFileSpec()) ||
      module_spec.GetUUID().IsValid()) {
    if (resolved_module_spec.GetArchitecture().IsValid() ||
        resolved_module_spec.GetUUID().IsValid()) {
      error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                          module_search_paths_ptr, nullptr,
                                          nullptr);

      if (exe_module_sp && exe_module_sp->GetObjectFile())
        return error;
      exe_module_sp.reset();
    }
    // No valid architecture was specified or the exact arch wasn't found so
    // ask the platform for the architectures that we should be using (in the
    // correct order) and see if we can find a match that way
    StreamString arch_names;
    llvm::ListSeparator LS;
    for (const ArchSpec &arch : GetSupportedArchitectures()) {
      resolved_module_spec.GetArchitecture() = arch;
      error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                          module_search_paths_ptr, nullptr,
                                          nullptr);
      // Did we find an executable using one of the
      if (error.Success()) {
        if (exe_module_sp && exe_module_sp->GetObjectFile())
          break;
        else
          error.SetErrorToGenericError();
      }

      arch_names << LS << arch.GetArchitectureName();
    }

    if (error.Fail() || !exe_module_sp) {
      if (FileSystem::Instance().Readable(resolved_module_spec.GetFileSpec())) {
        error.SetErrorStringWithFormatv(
            "'{0}' doesn't contain any '{1}' platform architectures: {2}",
            resolved_module_spec.GetFileSpec(), GetPluginName(),
            arch_names.GetData());
      } else {
        error.SetErrorStringWithFormatv("'{0}' is not readable",
                                        resolved_module_spec.GetFileSpec());
      }
    }
  } else {
    error.SetErrorStringWithFormatv("'{0}' does not exist",
                                    resolved_module_spec.GetFileSpec());
  }

  return error;
}

Status Platform::ResolveSymbolFile(Target &target, const ModuleSpec &sym_spec,
                                   FileSpec &sym_file) {
  Status error;
  if (FileSystem::Instance().Exists(sym_spec.GetSymbolFileSpec()))
    sym_file = sym_spec.GetSymbolFileSpec();
  else
    error.SetErrorString("unable to resolve symbol file");
  return error;
}

bool Platform::ResolveRemotePath(const FileSpec &platform_path,
                                 FileSpec &resolved_platform_path) {
  resolved_platform_path = platform_path;
  FileSystem::Instance().Resolve(resolved_platform_path);
  return true;
}

const ArchSpec &Platform::GetSystemArchitecture() {
  if (IsHost()) {
    if (!m_system_arch.IsValid()) {
      // We have a local host platform
      m_system_arch = HostInfo::GetArchitecture();
      m_system_arch_set_while_connected = m_system_arch.IsValid();
    }
  } else {
    // We have a remote platform. We can only fetch the remote system
    // architecture if we are connected, and we don't want to do it more than
    // once.

    const bool is_connected = IsConnected();

    bool fetch = false;
    if (m_system_arch.IsValid()) {
      // We have valid OS version info, check to make sure it wasn't manually
      // set prior to connecting. If it was manually set prior to connecting,
      // then lets fetch the actual OS version info if we are now connected.
      if (is_connected && !m_system_arch_set_while_connected)
        fetch = true;
    } else {
      // We don't have valid OS version info, fetch it if we are connected
      fetch = is_connected;
    }

    if (fetch) {
      m_system_arch = GetRemoteSystemArchitecture();
      m_system_arch_set_while_connected = m_system_arch.IsValid();
    }
  }
  return m_system_arch;
}

ArchSpec Platform::GetAugmentedArchSpec(llvm::StringRef triple) {
  if (triple.empty())
    return ArchSpec();
  llvm::Triple normalized_triple(llvm::Triple::normalize(triple));
  if (!ArchSpec::ContainsOnlyArch(normalized_triple))
    return ArchSpec(triple);

  if (auto kind = HostInfo::ParseArchitectureKind(triple))
    return HostInfo::GetArchitecture(*kind);

  ArchSpec compatible_arch;
  ArchSpec raw_arch(triple);
  if (!IsCompatibleArchitecture(raw_arch, false, &compatible_arch))
    return raw_arch;

  if (!compatible_arch.IsValid())
    return ArchSpec(normalized_triple);

  const llvm::Triple &compatible_triple = compatible_arch.GetTriple();
  if (normalized_triple.getVendorName().empty())
    normalized_triple.setVendor(compatible_triple.getVendor());
  if (normalized_triple.getOSName().empty())
    normalized_triple.setOS(compatible_triple.getOS());
  if (normalized_triple.getEnvironmentName().empty())
    normalized_triple.setEnvironment(compatible_triple.getEnvironment());
  return ArchSpec(normalized_triple);
}

Status Platform::ConnectRemote(Args &args) {
  Status error;
  if (IsHost())
    error.SetErrorStringWithFormatv(
        "The currently selected platform ({0}) is "
        "the host platform and is always connected.",
        GetPluginName());
  else
    error.SetErrorStringWithFormatv(
        "Platform::ConnectRemote() is not supported by {0}", GetPluginName());
  return error;
}

Status Platform::DisconnectRemote() {
  Status error;
  if (IsHost())
    error.SetErrorStringWithFormatv(
        "The currently selected platform ({0}) is "
        "the host platform and is always connected.",
        GetPluginName());
  else
    error.SetErrorStringWithFormatv(
        "Platform::DisconnectRemote() is not supported by {0}",
        GetPluginName());
  return error;
}

bool Platform::GetProcessInfo(lldb::pid_t pid,
                              ProcessInstanceInfo &process_info) {
  // Take care of the host case so that each subclass can just call this
  // function to get the host functionality.
  if (IsHost())
    return Host::GetProcessInfo(pid, process_info);
  return false;
}

uint32_t Platform::FindProcesses(const ProcessInstanceInfoMatch &match_info,
                                 ProcessInstanceInfoList &process_infos) {
  // Take care of the host case so that each subclass can just call this
  // function to get the host functionality.
  uint32_t match_count = 0;
  if (IsHost())
    match_count = Host::FindProcesses(match_info, process_infos);
  return match_count;
}

Status Platform::LaunchProcess(ProcessLaunchInfo &launch_info) {
  Status error;
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  LLDB_LOGF(log, "Platform::%s()", __FUNCTION__);

  // Take care of the host case so that each subclass can just call this
  // function to get the host functionality.
  if (IsHost()) {
    if (::getenv("LLDB_LAUNCH_FLAG_LAUNCH_IN_TTY"))
      launch_info.GetFlags().Set(eLaunchFlagLaunchInTTY);

    if (launch_info.GetFlags().Test(eLaunchFlagLaunchInShell)) {
      const bool will_debug = launch_info.GetFlags().Test(eLaunchFlagDebug);
      const bool first_arg_is_full_shell_command = false;
      uint32_t num_resumes = GetResumeCountForLaunchInfo(launch_info);
      if (log) {
        const FileSpec &shell = launch_info.GetShell();
        std::string shell_str = (shell) ? shell.GetPath() : "<null>";
        LLDB_LOGF(log,
                  "Platform::%s GetResumeCountForLaunchInfo() returned %" PRIu32
                  ", shell is '%s'",
                  __FUNCTION__, num_resumes, shell_str.c_str());
      }

      if (!launch_info.ConvertArgumentsForLaunchingInShell(
              error, will_debug, first_arg_is_full_shell_command, num_resumes))
        return error;
    } else if (launch_info.GetFlags().Test(eLaunchFlagShellExpandArguments)) {
      error = ShellExpandArguments(launch_info);
      if (error.Fail()) {
        error.SetErrorStringWithFormat("shell expansion failed (reason: %s). "
                                       "consider launching with 'process "
                                       "launch'.",
                                       error.AsCString("unknown"));
        return error;
      }
    }

    LLDB_LOGF(log, "Platform::%s final launch_info resume count: %" PRIu32,
              __FUNCTION__, launch_info.GetResumeCount());

    error = Host::LaunchProcess(launch_info);
  } else
    error.SetErrorString(
        "base lldb_private::Platform class can't launch remote processes");
  return error;
}

Status Platform::ShellExpandArguments(ProcessLaunchInfo &launch_info) {
  if (IsHost())
    return Host::ShellExpandArguments(launch_info);
  return Status("base lldb_private::Platform class can't expand arguments");
}

Status Platform::KillProcess(const lldb::pid_t pid) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  LLDB_LOGF(log, "Platform::%s, pid %" PRIu64, __FUNCTION__, pid);

  if (!IsHost()) {
    return Status(
        "base lldb_private::Platform class can't kill remote processes");
  }
  Host::Kill(pid, SIGKILL);
  return Status();
}

lldb::ProcessSP Platform::DebugProcess(ProcessLaunchInfo &launch_info,
                                       Debugger &debugger, Target &target,
                                       Status &error) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  LLDB_LOG(log, "target = {0})", &target);

  ProcessSP process_sp;
  // Make sure we stop at the entry point
  launch_info.GetFlags().Set(eLaunchFlagDebug);
  // We always launch the process we are going to debug in a separate process
  // group, since then we can handle ^C interrupts ourselves w/o having to
  // worry about the target getting them as well.
  launch_info.SetLaunchInSeparateProcessGroup(true);

  // Allow any StructuredData process-bound plugins to adjust the launch info
  // if needed
  size_t i = 0;
  bool iteration_complete = false;
  // Note iteration can't simply go until a nullptr callback is returned, as it
  // is valid for a plugin to not supply a filter.
  auto get_filter_func = PluginManager::GetStructuredDataFilterCallbackAtIndex;
  for (auto filter_callback = get_filter_func(i, iteration_complete);
       !iteration_complete;
       filter_callback = get_filter_func(++i, iteration_complete)) {
    if (filter_callback) {
      // Give this ProcessLaunchInfo filter a chance to adjust the launch info.
      error = (*filter_callback)(launch_info, &target);
      if (!error.Success()) {
        LLDB_LOGF(log,
                  "Platform::%s() StructuredDataPlugin launch "
                  "filter failed.",
                  __FUNCTION__);
        return process_sp;
      }
    }
  }

  error = LaunchProcess(launch_info);
  if (error.Success()) {
    LLDB_LOGF(log,
              "Platform::%s LaunchProcess() call succeeded (pid=%" PRIu64 ")",
              __FUNCTION__, launch_info.GetProcessID());
    if (launch_info.GetProcessID() != LLDB_INVALID_PROCESS_ID) {
      ProcessAttachInfo attach_info(launch_info);
      process_sp = Attach(attach_info, debugger, &target, error);
      if (process_sp) {
        LLDB_LOG(log, "Attach() succeeded, Process plugin: {0}",
                 process_sp->GetPluginName());
        launch_info.SetHijackListener(attach_info.GetHijackListener());

        // Since we attached to the process, it will think it needs to detach
        // if the process object just goes away without an explicit call to
        // Process::Kill() or Process::Detach(), so let it know to kill the
        // process if this happens.
        process_sp->SetShouldDetach(false);

        // If we didn't have any file actions, the pseudo terminal might have
        // been used where the secondary side was given as the file to open for
        // stdin/out/err after we have already opened the primary so we can
        // read/write stdin/out/err.
        int pty_fd = launch_info.GetPTY().ReleasePrimaryFileDescriptor();
        if (pty_fd != PseudoTerminal::invalid_fd) {
          process_sp->SetSTDIOFileDescriptor(pty_fd);
        }
      } else {
        LLDB_LOGF(log, "Platform::%s Attach() failed: %s", __FUNCTION__,
                  error.AsCString());
      }
    } else {
      LLDB_LOGF(log,
                "Platform::%s LaunchProcess() returned launch_info with "
                "invalid process id",
                __FUNCTION__);
    }
  } else {
    LLDB_LOGF(log, "Platform::%s LaunchProcess() failed: %s", __FUNCTION__,
              error.AsCString());
  }

  return process_sp;
}

lldb::PlatformSP
Platform::GetPlatformForArchitecture(const ArchSpec &arch,
                                     ArchSpec *platform_arch_ptr) {
  lldb::PlatformSP platform_sp;
  Status error;
  if (arch.IsValid())
    platform_sp = Platform::Create(arch, platform_arch_ptr, error);
  return platform_sp;
}

std::vector<ArchSpec>
Platform::CreateArchList(llvm::ArrayRef<llvm::Triple::ArchType> archs,
                         llvm::Triple::OSType os) {
  std::vector<ArchSpec> list;
  for(auto arch : archs) {
    llvm::Triple triple;
    triple.setArch(arch);
    triple.setOS(os);
    list.push_back(ArchSpec(triple));
  }
  return list;
}

/// Lets a platform answer if it is compatible with a given
/// architecture and the target triple contained within.
bool Platform::IsCompatibleArchitecture(const ArchSpec &arch,
                                        bool exact_arch_match,
                                        ArchSpec *compatible_arch_ptr) {
  // If the architecture is invalid, we must answer true...
  if (arch.IsValid()) {
    ArchSpec platform_arch;
    auto match = exact_arch_match ? &ArchSpec::IsExactMatch
                                  : &ArchSpec::IsCompatibleMatch;
    for (const ArchSpec &platform_arch : GetSupportedArchitectures()) {
      if ((arch.*match)(platform_arch)) {
        if (compatible_arch_ptr)
          *compatible_arch_ptr = platform_arch;
        return true;
      }
    }
  }
  if (compatible_arch_ptr)
    compatible_arch_ptr->Clear();
  return false;
}

Status Platform::PutFile(const FileSpec &source, const FileSpec &destination,
                         uint32_t uid, uint32_t gid) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  LLDB_LOGF(log, "[PutFile] Using block by block transfer....\n");

  auto source_open_options =
      File::eOpenOptionReadOnly | File::eOpenOptionCloseOnExec;
  namespace fs = llvm::sys::fs;
  if (fs::is_symlink_file(source.GetPath()))
    source_open_options |= File::eOpenOptionDontFollowSymlinks;

  auto source_file = FileSystem::Instance().Open(source, source_open_options,
                                                 lldb::eFilePermissionsUserRW);
  if (!source_file)
    return Status(source_file.takeError());
  Status error;
  uint32_t permissions = source_file.get()->GetPermissions(error);
  if (permissions == 0)
    permissions = lldb::eFilePermissionsFileDefault;

  lldb::user_id_t dest_file = OpenFile(
      destination, File::eOpenOptionCanCreate | File::eOpenOptionWriteOnly |
                       File::eOpenOptionTruncate | File::eOpenOptionCloseOnExec,
      permissions, error);
  LLDB_LOGF(log, "dest_file = %" PRIu64 "\n", dest_file);

  if (error.Fail())
    return error;
  if (dest_file == UINT64_MAX)
    return Status("unable to open target file");
  lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024 * 16, 0));
  uint64_t offset = 0;
  for (;;) {
    size_t bytes_read = buffer_sp->GetByteSize();
    error = source_file.get()->Read(buffer_sp->GetBytes(), bytes_read);
    if (error.Fail() || bytes_read == 0)
      break;

    const uint64_t bytes_written =
        WriteFile(dest_file, offset, buffer_sp->GetBytes(), bytes_read, error);
    if (error.Fail())
      break;

    offset += bytes_written;
    if (bytes_written != bytes_read) {
      // We didn't write the correct number of bytes, so adjust the file
      // position in the source file we are reading from...
      source_file.get()->SeekFromStart(offset);
    }
  }
  CloseFile(dest_file, error);

  if (uid == UINT32_MAX && gid == UINT32_MAX)
    return error;

  // TODO: ChownFile?

  return error;
}

Status Platform::GetFile(const FileSpec &source, const FileSpec &destination) {
  Status error("unimplemented");
  return error;
}

Status
Platform::CreateSymlink(const FileSpec &src, // The name of the link is in src
                        const FileSpec &dst) // The symlink points to dst
{
  Status error("unimplemented");
  return error;
}

bool Platform::GetFileExists(const lldb_private::FileSpec &file_spec) {
  return false;
}

Status Platform::Unlink(const FileSpec &path) {
  Status error("unimplemented");
  return error;
}

MmapArgList Platform::GetMmapArgumentList(const ArchSpec &arch, addr_t addr,
                                          addr_t length, unsigned prot,
                                          unsigned flags, addr_t fd,
                                          addr_t offset) {
  uint64_t flags_platform = 0;
  if (flags & eMmapFlagsPrivate)
    flags_platform |= MAP_PRIVATE;
  if (flags & eMmapFlagsAnon)
    flags_platform |= MAP_ANON;

  MmapArgList args({addr, length, prot, flags_platform, fd, offset});
  return args;
}

lldb_private::Status Platform::RunShellCommand(
    llvm::StringRef command,
    const FileSpec &
        working_dir, // Pass empty FileSpec to use the current working directory
    int *status_ptr, // Pass nullptr if you don't want the process exit status
    int *signo_ptr, // Pass nullptr if you don't want the signal that caused the
                    // process to exit
    std::string
        *command_output, // Pass nullptr if you don't want the command output
    const Timeout<std::micro> &timeout) {
  return RunShellCommand(llvm::StringRef(), command, working_dir, status_ptr,
                         signo_ptr, command_output, timeout);
}

lldb_private::Status Platform::RunShellCommand(
    llvm::StringRef shell,   // Pass empty if you want to use the default
                             // shell interpreter
    llvm::StringRef command, // Shouldn't be empty
    const FileSpec &
        working_dir, // Pass empty FileSpec to use the current working directory
    int *status_ptr, // Pass nullptr if you don't want the process exit status
    int *signo_ptr, // Pass nullptr if you don't want the signal that caused the
                    // process to exit
    std::string
        *command_output, // Pass nullptr if you don't want the command output
    const Timeout<std::micro> &timeout) {
  if (IsHost())
    return Host::RunShellCommand(shell, command, working_dir, status_ptr,
                                 signo_ptr, command_output, timeout);
  else
    return Status("unimplemented");
}

bool Platform::CalculateMD5(const FileSpec &file_spec, uint64_t &low,
                            uint64_t &high) {
  if (!IsHost())
    return false;
  auto Result = llvm::sys::fs::md5_contents(file_spec.GetPath());
  if (!Result)
    return false;
  std::tie(high, low) = Result->words();
  return true;
}

void Platform::SetLocalCacheDirectory(const char *local) {
  m_local_cache_directory.assign(local);
}

const char *Platform::GetLocalCacheDirectory() {
  return m_local_cache_directory.c_str();
}

static constexpr OptionDefinition g_rsync_option_table[] = {
    {LLDB_OPT_SET_ALL, false, "rsync", 'r', OptionParser::eNoArgument, nullptr,
     {}, 0, eArgTypeNone, "Enable rsync."},
    {LLDB_OPT_SET_ALL, false, "rsync-opts", 'R',
     OptionParser::eRequiredArgument, nullptr, {}, 0, eArgTypeCommandName,
     "Platform-specific options required for rsync to work."},
    {LLDB_OPT_SET_ALL, false, "rsync-prefix", 'P',
     OptionParser::eRequiredArgument, nullptr, {}, 0, eArgTypeCommandName,
     "Platform-specific rsync prefix put before the remote path."},
    {LLDB_OPT_SET_ALL, false, "ignore-remote-hostname", 'i',
     OptionParser::eNoArgument, nullptr, {}, 0, eArgTypeNone,
     "Do not automatically fill in the remote hostname when composing the "
     "rsync command."},
};

static constexpr OptionDefinition g_ssh_option_table[] = {
    {LLDB_OPT_SET_ALL, false, "ssh", 's', OptionParser::eNoArgument, nullptr,
     {}, 0, eArgTypeNone, "Enable SSH."},
    {LLDB_OPT_SET_ALL, false, "ssh-opts", 'S', OptionParser::eRequiredArgument,
     nullptr, {}, 0, eArgTypeCommandName,
     "Platform-specific options required for SSH to work."},
};

static constexpr OptionDefinition g_caching_option_table[] = {
    {LLDB_OPT_SET_ALL, false, "local-cache-dir", 'c',
     OptionParser::eRequiredArgument, nullptr, {}, 0, eArgTypePath,
     "Path in which to store local copies of files."},
};

llvm::ArrayRef<OptionDefinition> OptionGroupPlatformRSync::GetDefinitions() {
  return llvm::makeArrayRef(g_rsync_option_table);
}

void OptionGroupPlatformRSync::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_rsync = false;
  m_rsync_opts.clear();
  m_rsync_prefix.clear();
  m_ignores_remote_hostname = false;
}

lldb_private::Status
OptionGroupPlatformRSync::SetOptionValue(uint32_t option_idx,
                                         llvm::StringRef option_arg,
                                         ExecutionContext *execution_context) {
  Status error;
  char short_option = (char)GetDefinitions()[option_idx].short_option;
  switch (short_option) {
  case 'r':
    m_rsync = true;
    break;

  case 'R':
    m_rsync_opts.assign(std::string(option_arg));
    break;

  case 'P':
    m_rsync_prefix.assign(std::string(option_arg));
    break;

  case 'i':
    m_ignores_remote_hostname = true;
    break;

  default:
    error.SetErrorStringWithFormat("unrecognized option '%c'", short_option);
    break;
  }

  return error;
}

lldb::BreakpointSP
Platform::SetThreadCreationBreakpoint(lldb_private::Target &target) {
  return lldb::BreakpointSP();
}

llvm::ArrayRef<OptionDefinition> OptionGroupPlatformSSH::GetDefinitions() {
  return llvm::makeArrayRef(g_ssh_option_table);
}

void OptionGroupPlatformSSH::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_ssh = false;
  m_ssh_opts.clear();
}

lldb_private::Status
OptionGroupPlatformSSH::SetOptionValue(uint32_t option_idx,
                                       llvm::StringRef option_arg,
                                       ExecutionContext *execution_context) {
  Status error;
  char short_option = (char)GetDefinitions()[option_idx].short_option;
  switch (short_option) {
  case 's':
    m_ssh = true;
    break;

  case 'S':
    m_ssh_opts.assign(std::string(option_arg));
    break;

  default:
    error.SetErrorStringWithFormat("unrecognized option '%c'", short_option);
    break;
  }

  return error;
}

llvm::ArrayRef<OptionDefinition> OptionGroupPlatformCaching::GetDefinitions() {
  return llvm::makeArrayRef(g_caching_option_table);
}

void OptionGroupPlatformCaching::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_cache_dir.clear();
}

lldb_private::Status OptionGroupPlatformCaching::SetOptionValue(
    uint32_t option_idx, llvm::StringRef option_arg,
    ExecutionContext *execution_context) {
  Status error;
  char short_option = (char)GetDefinitions()[option_idx].short_option;
  switch (short_option) {
  case 'c':
    m_cache_dir.assign(std::string(option_arg));
    break;

  default:
    error.SetErrorStringWithFormat("unrecognized option '%c'", short_option);
    break;
  }

  return error;
}

Environment Platform::GetEnvironment() { return Environment(); }

const std::vector<ConstString> &Platform::GetTrapHandlerSymbolNames() {
  if (!m_calculated_trap_handlers) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_calculated_trap_handlers) {
      CalculateTrapHandlerSymbolNames();
      m_calculated_trap_handlers = true;
    }
  }
  return m_trap_handlers;
}

Status
Platform::GetCachedExecutable(ModuleSpec &module_spec,
                              lldb::ModuleSP &module_sp,
                              const FileSpecList *module_search_paths_ptr) {
  FileSpec platform_spec = module_spec.GetFileSpec();
  Status error = GetRemoteSharedModule(
      module_spec, nullptr, module_sp,
      [&](const ModuleSpec &spec) {
        return ResolveRemoteExecutable(spec, module_sp,
                                       module_search_paths_ptr);
      },
      nullptr);
  if (error.Success()) {
    module_spec.GetFileSpec() = module_sp->GetFileSpec();
    module_spec.GetPlatformFileSpec() = platform_spec;
  }

  return error;
}

Status Platform::GetRemoteSharedModule(const ModuleSpec &module_spec,
                                       Process *process,
                                       lldb::ModuleSP &module_sp,
                                       const ModuleResolver &module_resolver,
                                       bool *did_create_ptr) {
  // Get module information from a target.
  ModuleSpec resolved_module_spec;
  bool got_module_spec = false;
  if (process) {
    // Try to get module information from the process
    if (process->GetModuleSpec(module_spec.GetFileSpec(),
                               module_spec.GetArchitecture(),
                               resolved_module_spec)) {
      if (!module_spec.GetUUID().IsValid() ||
          module_spec.GetUUID() == resolved_module_spec.GetUUID()) {
        got_module_spec = true;
      }
    }
  }

  if (!module_spec.GetArchitecture().IsValid()) {
    Status error;
    // No valid architecture was specified, ask the platform for the
    // architectures that we should be using (in the correct order) and see if
    // we can find a match that way
    ModuleSpec arch_module_spec(module_spec);
    for (const ArchSpec &arch : GetSupportedArchitectures()) {
      arch_module_spec.GetArchitecture() = arch;
      error = ModuleList::GetSharedModule(arch_module_spec, module_sp, nullptr,
                                          nullptr, nullptr);
      // Did we find an executable using one of the
      if (error.Success() && module_sp)
        break;
    }
    if (module_sp) {
      resolved_module_spec = arch_module_spec;
      got_module_spec = true;
    }
  }

  if (!got_module_spec) {
    // Get module information from a target.
    if (GetModuleSpec(module_spec.GetFileSpec(), module_spec.GetArchitecture(),
                      resolved_module_spec)) {
      if (!module_spec.GetUUID().IsValid() ||
          module_spec.GetUUID() == resolved_module_spec.GetUUID()) {
        got_module_spec = true;
      }
    }
  }

  if (!got_module_spec) {
    // Fall back to the given module resolver, which may have its own
    // search logic.
    return module_resolver(module_spec);
  }

  // If we are looking for a specific UUID, make sure resolved_module_spec has
  // the same one before we search.
  if (module_spec.GetUUID().IsValid()) {
    resolved_module_spec.GetUUID() = module_spec.GetUUID();
  }

  // Trying to find a module by UUID on local file system.
  const auto error = module_resolver(resolved_module_spec);
  if (error.Fail()) {
    if (GetCachedSharedModule(resolved_module_spec, module_sp, did_create_ptr))
      return Status();
  }

  return error;
}

bool Platform::GetCachedSharedModule(const ModuleSpec &module_spec,
                                     lldb::ModuleSP &module_sp,
                                     bool *did_create_ptr) {
  if (IsHost() || !GetGlobalPlatformProperties().GetUseModuleCache() ||
      !GetGlobalPlatformProperties().GetModuleCacheDirectory())
    return false;

  Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);

  // Check local cache for a module.
  auto error = m_module_cache->GetAndPut(
      GetModuleCacheRoot(), GetCacheHostname(), module_spec,
      [this](const ModuleSpec &module_spec,
             const FileSpec &tmp_download_file_spec) {
        return DownloadModuleSlice(
            module_spec.GetFileSpec(), module_spec.GetObjectOffset(),
            module_spec.GetObjectSize(), tmp_download_file_spec);

      },
      [this](const ModuleSP &module_sp,
             const FileSpec &tmp_download_file_spec) {
        return DownloadSymbolFile(module_sp, tmp_download_file_spec);
      },
      module_sp, did_create_ptr);
  if (error.Success())
    return true;

  LLDB_LOGF(log, "Platform::%s - module %s not found in local cache: %s",
            __FUNCTION__, module_spec.GetUUID().GetAsString().c_str(),
            error.AsCString());
  return false;
}

Status Platform::DownloadModuleSlice(const FileSpec &src_file_spec,
                                     const uint64_t src_offset,
                                     const uint64_t src_size,
                                     const FileSpec &dst_file_spec) {
  Status error;

  std::error_code EC;
  llvm::raw_fd_ostream dst(dst_file_spec.GetPath(), EC, llvm::sys::fs::OF_None);
  if (EC) {
    error.SetErrorStringWithFormat("unable to open destination file: %s",
                                   dst_file_spec.GetPath().c_str());
    return error;
  }

  auto src_fd = OpenFile(src_file_spec, File::eOpenOptionReadOnly,
                         lldb::eFilePermissionsFileDefault, error);

  if (error.Fail()) {
    error.SetErrorStringWithFormat("unable to open source file: %s",
                                   error.AsCString());
    return error;
  }

  std::vector<char> buffer(1024);
  auto offset = src_offset;
  uint64_t total_bytes_read = 0;
  while (total_bytes_read < src_size) {
    const auto to_read = std::min(static_cast<uint64_t>(buffer.size()),
                                  src_size - total_bytes_read);
    const uint64_t n_read =
        ReadFile(src_fd, offset, &buffer[0], to_read, error);
    if (error.Fail())
      break;
    if (n_read == 0) {
      error.SetErrorString("read 0 bytes");
      break;
    }
    offset += n_read;
    total_bytes_read += n_read;
    dst.write(&buffer[0], n_read);
  }

  Status close_error;
  CloseFile(src_fd, close_error); // Ignoring close error.

  return error;
}

Status Platform::DownloadSymbolFile(const lldb::ModuleSP &module_sp,
                                    const FileSpec &dst_file_spec) {
  return Status(
      "Symbol file downloading not supported by the default platform.");
}

FileSpec Platform::GetModuleCacheRoot() {
  auto dir_spec = GetGlobalPlatformProperties().GetModuleCacheDirectory();
  dir_spec.AppendPathComponent(GetName().AsCString());
  return dir_spec;
}

const char *Platform::GetCacheHostname() { return GetHostname(); }

const UnixSignalsSP &Platform::GetRemoteUnixSignals() {
  static const auto s_default_unix_signals_sp = std::make_shared<UnixSignals>();
  return s_default_unix_signals_sp;
}

UnixSignalsSP Platform::GetUnixSignals() {
  if (IsHost())
    return UnixSignals::CreateForHost();
  return GetRemoteUnixSignals();
}

uint32_t Platform::LoadImage(lldb_private::Process *process,
                             const lldb_private::FileSpec &local_file,
                             const lldb_private::FileSpec &remote_file,
                             lldb_private::Status &error) {
  if (local_file && remote_file) {
    // Both local and remote file was specified. Install the local file to the
    // given location.
    if (IsRemote() || local_file != remote_file) {
      error = Install(local_file, remote_file);
      if (error.Fail())
        return LLDB_INVALID_IMAGE_TOKEN;
    }
    return DoLoadImage(process, remote_file, nullptr, error);
  }

  if (local_file) {
    // Only local file was specified. Install it to the current working
    // directory.
    FileSpec target_file = GetWorkingDirectory();
    target_file.AppendPathComponent(local_file.GetFilename().AsCString());
    if (IsRemote() || local_file != target_file) {
      error = Install(local_file, target_file);
      if (error.Fail())
        return LLDB_INVALID_IMAGE_TOKEN;
    }
    return DoLoadImage(process, target_file, nullptr, error);
  }

  if (remote_file) {
    // Only remote file was specified so we don't have to do any copying
    return DoLoadImage(process, remote_file, nullptr, error);
  }

  error.SetErrorString("Neither local nor remote file was specified");
  return LLDB_INVALID_IMAGE_TOKEN;
}

uint32_t Platform::DoLoadImage(lldb_private::Process *process,
                               const lldb_private::FileSpec &remote_file,
                               const std::vector<std::string> *paths,
                               lldb_private::Status &error,
                               lldb_private::FileSpec *loaded_image) {
  error.SetErrorString("LoadImage is not supported on the current platform");
  return LLDB_INVALID_IMAGE_TOKEN;
}

uint32_t Platform::LoadImageUsingPaths(lldb_private::Process *process,
                               const lldb_private::FileSpec &remote_filename,
                               const std::vector<std::string> &paths,
                               lldb_private::Status &error,
                               lldb_private::FileSpec *loaded_path)
{
  FileSpec file_to_use;
  if (remote_filename.IsAbsolute())
    file_to_use = FileSpec(remote_filename.GetFilename().GetStringRef(),

                           remote_filename.GetPathStyle());
  else
    file_to_use = remote_filename;

  return DoLoadImage(process, file_to_use, &paths, error, loaded_path);
}

Status Platform::UnloadImage(lldb_private::Process *process,
                             uint32_t image_token) {
  return Status("UnloadImage is not supported on the current platform");
}

lldb::ProcessSP Platform::ConnectProcess(llvm::StringRef connect_url,
                                         llvm::StringRef plugin_name,
                                         Debugger &debugger, Target *target,
                                         Status &error) {
  return DoConnectProcess(connect_url, plugin_name, debugger, nullptr, target,
                          error);
}

lldb::ProcessSP Platform::ConnectProcessSynchronous(
    llvm::StringRef connect_url, llvm::StringRef plugin_name,
    Debugger &debugger, Stream &stream, Target *target, Status &error) {
  return DoConnectProcess(connect_url, plugin_name, debugger, &stream, target,
                          error);
}

lldb::ProcessSP Platform::DoConnectProcess(llvm::StringRef connect_url,
                                           llvm::StringRef plugin_name,
                                           Debugger &debugger, Stream *stream,
                                           Target *target, Status &error) {
  error.Clear();

  if (!target) {
    ArchSpec arch;
    if (target && target->GetArchitecture().IsValid())
      arch = target->GetArchitecture();
    else
      arch = Target::GetDefaultArchitecture();

    const char *triple = "";
    if (arch.IsValid())
      triple = arch.GetTriple().getTriple().c_str();

    TargetSP new_target_sp;
    error = debugger.GetTargetList().CreateTarget(
        debugger, "", triple, eLoadDependentsNo, nullptr, new_target_sp);
    target = new_target_sp.get();
  }

  if (!target || error.Fail())
    return nullptr;

  lldb::ProcessSP process_sp =
      target->CreateProcess(debugger.GetListener(), plugin_name, nullptr, true);

  if (!process_sp)
    return nullptr;

  // If this private method is called with a stream we are synchronous.
  const bool synchronous = stream != nullptr;

  ListenerSP listener_sp(
      Listener::MakeListener("lldb.Process.ConnectProcess.hijack"));
  if (synchronous)
    process_sp->HijackProcessEvents(listener_sp);

  error = process_sp->ConnectRemote(connect_url);
  if (error.Fail()) {
    if (synchronous)
      process_sp->RestoreProcessEvents();
    return nullptr;
  }

  if (synchronous) {
    EventSP event_sp;
    process_sp->WaitForProcessToStop(llvm::None, &event_sp, true, listener_sp,
                                     nullptr);
    process_sp->RestoreProcessEvents();
    bool pop_process_io_handler = false;
    Process::HandleProcessStateChangedEvent(event_sp, stream,
                                            pop_process_io_handler);
  }

  return process_sp;
}

size_t Platform::ConnectToWaitingProcesses(lldb_private::Debugger &debugger,
                                           lldb_private::Status &error) {
  error.Clear();
  return 0;
}

size_t Platform::GetSoftwareBreakpointTrapOpcode(Target &target,
                                                 BreakpointSite *bp_site) {
  ArchSpec arch = target.GetArchitecture();
  assert(arch.IsValid());
  const uint8_t *trap_opcode = nullptr;
  size_t trap_opcode_size = 0;

  switch (arch.GetMachine()) {
  case llvm::Triple::aarch64_32:
  case llvm::Triple::aarch64: {
    static const uint8_t g_aarch64_opcode[] = {0x00, 0x00, 0x20, 0xd4};
    trap_opcode = g_aarch64_opcode;
    trap_opcode_size = sizeof(g_aarch64_opcode);
  } break;

  case llvm::Triple::arc: {
    static const uint8_t g_hex_opcode[] = { 0xff, 0x7f };
    trap_opcode = g_hex_opcode;
    trap_opcode_size = sizeof(g_hex_opcode);
  } break;

  // TODO: support big-endian arm and thumb trap codes.
  case llvm::Triple::arm: {
    // The ARM reference recommends the use of 0xe7fddefe and 0xdefe but the
    // linux kernel does otherwise.
    static const uint8_t g_arm_breakpoint_opcode[] = {0xf0, 0x01, 0xf0, 0xe7};
    static const uint8_t g_thumb_breakpoint_opcode[] = {0x01, 0xde};

    lldb::BreakpointLocationSP bp_loc_sp(bp_site->GetOwnerAtIndex(0));
    AddressClass addr_class = AddressClass::eUnknown;

    if (bp_loc_sp) {
      addr_class = bp_loc_sp->GetAddress().GetAddressClass();
      if (addr_class == AddressClass::eUnknown &&
          (bp_loc_sp->GetAddress().GetFileAddress() & 1))
        addr_class = AddressClass::eCodeAlternateISA;
    }

    if (addr_class == AddressClass::eCodeAlternateISA) {
      trap_opcode = g_thumb_breakpoint_opcode;
      trap_opcode_size = sizeof(g_thumb_breakpoint_opcode);
    } else {
      trap_opcode = g_arm_breakpoint_opcode;
      trap_opcode_size = sizeof(g_arm_breakpoint_opcode);
    }
  } break;

  case llvm::Triple::avr: {
    static const uint8_t g_hex_opcode[] = {0x98, 0x95};
    trap_opcode = g_hex_opcode;
    trap_opcode_size = sizeof(g_hex_opcode);
  } break;

  case llvm::Triple::mips:
  case llvm::Triple::mips64: {
    static const uint8_t g_hex_opcode[] = {0x00, 0x00, 0x00, 0x0d};
    trap_opcode = g_hex_opcode;
    trap_opcode_size = sizeof(g_hex_opcode);
  } break;

  case llvm::Triple::mipsel:
  case llvm::Triple::mips64el: {
    static const uint8_t g_hex_opcode[] = {0x0d, 0x00, 0x00, 0x00};
    trap_opcode = g_hex_opcode;
    trap_opcode_size = sizeof(g_hex_opcode);
  } break;

  case llvm::Triple::systemz: {
    static const uint8_t g_hex_opcode[] = {0x00, 0x01};
    trap_opcode = g_hex_opcode;
    trap_opcode_size = sizeof(g_hex_opcode);
  } break;

  case llvm::Triple::hexagon: {
    static const uint8_t g_hex_opcode[] = {0x0c, 0xdb, 0x00, 0x54};
    trap_opcode = g_hex_opcode;
    trap_opcode_size = sizeof(g_hex_opcode);
  } break;

  case llvm::Triple::ppc:
  case llvm::Triple::ppc64: {
    static const uint8_t g_ppc_opcode[] = {0x7f, 0xe0, 0x00, 0x08};
    trap_opcode = g_ppc_opcode;
    trap_opcode_size = sizeof(g_ppc_opcode);
  } break;

  case llvm::Triple::ppc64le: {
    static const uint8_t g_ppc64le_opcode[] = {0x08, 0x00, 0xe0, 0x7f}; // trap
    trap_opcode = g_ppc64le_opcode;
    trap_opcode_size = sizeof(g_ppc64le_opcode);
  } break;

  case llvm::Triple::x86:
  case llvm::Triple::x86_64: {
    static const uint8_t g_i386_opcode[] = {0xCC};
    trap_opcode = g_i386_opcode;
    trap_opcode_size = sizeof(g_i386_opcode);
  } break;

  default:
    return 0;
  }

  assert(bp_site);
  if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
    return trap_opcode_size;

  return 0;
}
