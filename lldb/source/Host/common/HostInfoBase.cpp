//===-- HostInfoBase.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostInfoBase.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/Host.h"

#include <thread>

using namespace lldb;
using namespace lldb_private;

namespace
{
void
CleanupProcessSpecificLLDBTempDir()
{
    // Get the process specific LLDB temporary directory and delete it.
    FileSpec tmpdir_file_spec;
    if (!HostInfo::GetLLDBPath(ePathTypeLLDBTempSystemDir, tmpdir_file_spec))
        return;

    // Remove the LLDB temporary directory if we have one. Set "recurse" to
    // true to all files that were created for the LLDB process can be cleaned up.
    FileSystem::DeleteDirectory(tmpdir_file_spec.GetDirectory().GetCString(), true);
}
}

#define COMPUTE_LLDB_PATH(compute_function, member_var)                                                                                    \
    {                                                                                                                                      \
        static bool is_initialized = false;                                                                                                \
        static bool success = false;                                                                                                       \
        if (!is_initialized)                                                                                                               \
        {                                                                                                                                  \
            is_initialized = true;                                                                                                         \
            success = HostInfo::compute_function(member_var);                                                                              \
        }                                                                                                                                  \
        if (success)                                                                                                                       \
            result = &member_var;                                                                                                          \
    }

uint32_t HostInfoBase::m_number_cpus = 0;
std::string HostInfoBase::m_vendor_string;
std::string HostInfoBase::m_os_string;
std::string HostInfoBase::m_host_triple;
ArchSpec HostInfoBase::m_host_arch_32;
ArchSpec HostInfoBase::m_host_arch_64;

FileSpec HostInfoBase::m_lldb_so_dir;
FileSpec HostInfoBase::m_lldb_support_exe_dir;
FileSpec HostInfoBase::m_lldb_headers_dir;
FileSpec HostInfoBase::m_lldb_python_dir;
FileSpec HostInfoBase::m_lldb_system_plugin_dir;
FileSpec HostInfoBase::m_lldb_user_plugin_dir;
FileSpec HostInfoBase::m_lldb_tmp_dir;

uint32_t
HostInfoBase::GetNumberCPUS()
{
    static bool is_initialized = false;
    if (!is_initialized)
    {
        m_number_cpus = std::thread::hardware_concurrency();
        is_initialized = true;
    }

    return m_number_cpus;
}

llvm::StringRef
HostInfoBase::GetVendorString()
{
    static bool is_initialized = false;
    if (!is_initialized)
    {
        const ArchSpec &host_arch = HostInfo::GetArchitecture();
        const llvm::StringRef &str_ref = host_arch.GetTriple().getVendorName();
        m_vendor_string.assign(str_ref.begin(), str_ref.end());
        is_initialized = true;
    }
    return m_vendor_string;
}

llvm::StringRef
HostInfoBase::GetOSString()
{
    static bool is_initialized = false;
    if (!is_initialized)
    {
        const ArchSpec &host_arch = HostInfo::GetArchitecture();
        const llvm::StringRef &str_ref = host_arch.GetTriple().getOSName();
        m_os_string.assign(str_ref.begin(), str_ref.end());
        is_initialized = true;
    }
    return m_os_string;
}

llvm::StringRef
HostInfoBase::GetTargetTriple()
{
    static bool is_initialized = false;
    if (!is_initialized)
    {
        const ArchSpec &host_arch = HostInfo::GetArchitecture();
        m_host_triple = host_arch.GetTriple().getTriple();
        is_initialized = true;
    }
    return m_host_triple;
}

const ArchSpec &
HostInfoBase::GetArchitecture(ArchitectureKind arch_kind)
{
    static bool is_initialized = false;
    if (!is_initialized)
    {
        HostInfo::ComputeHostArchitectureSupport(m_host_arch_32, m_host_arch_64);
        is_initialized = true;
    }

    // If an explicit 32 or 64-bit architecture was requested, return that.
    if (arch_kind == eArchKind32)
        return m_host_arch_32;
    if (arch_kind == eArchKind64)
        return m_host_arch_64;

    // Otherwise prefer the 64-bit architecture if it is valid.
    return (m_host_arch_64.IsValid()) ? m_host_arch_64 : m_host_arch_32;
}

bool
HostInfoBase::GetLLDBPath(lldb::PathType type, FileSpec &file_spec)
{
    file_spec.Clear();

#if defined(LLDB_DISABLE_PYTHON)
    if (type == lldb::ePathTypePythonDir)
        return false;
#endif

    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
    FileSpec *result = nullptr;
    switch (type)
    {
        case lldb::ePathTypeLLDBShlibDir:
            COMPUTE_LLDB_PATH(ComputeSharedLibraryDirectory, m_lldb_so_dir)
            if (log)
                log->Printf("HostInfoBase::GetLLDBPath(ePathTypeLLDBShlibDir) => '%s'", m_lldb_so_dir.GetPath().c_str());
            break;
        case lldb::ePathTypeSupportExecutableDir:
            COMPUTE_LLDB_PATH(ComputeSupportExeDirectory, m_lldb_support_exe_dir)
            if (log)
                log->Printf("HostInfoBase::GetLLDBPath(ePathTypeSupportExecutableDir) => '%s'", m_lldb_support_exe_dir.GetPath().c_str());
            break;
        case lldb::ePathTypeHeaderDir:
            COMPUTE_LLDB_PATH(ComputeHeaderDirectory, m_lldb_headers_dir)
            if (log)
                log->Printf("HostInfoBase::GetLLDBPath(ePathTypeHeaderDir) => '%s'", m_lldb_headers_dir.GetPath().c_str());
            break;
        case lldb::ePathTypePythonDir:
            COMPUTE_LLDB_PATH(ComputePythonDirectory, m_lldb_python_dir)
            if (log)
                log->Printf("HostInfoBase::GetLLDBPath(ePathTypePythonDir) => '%s'", m_lldb_python_dir.GetPath().c_str());
            break;
        case lldb::ePathTypeLLDBSystemPlugins:
            COMPUTE_LLDB_PATH(ComputeSystemPluginsDirectory, m_lldb_system_plugin_dir)
            if (log)
                log->Printf("HostInfoBase::GetLLDBPath(ePathTypeLLDBSystemPlugins) => '%s'", m_lldb_system_plugin_dir.GetPath().c_str());
            break;
        case lldb::ePathTypeLLDBUserPlugins:
            COMPUTE_LLDB_PATH(ComputeUserPluginsDirectory, m_lldb_user_plugin_dir)
            if (log)
                log->Printf("HostInfoBase::GetLLDBPath(ePathTypeLLDBUserPlugins) => '%s'", m_lldb_user_plugin_dir.GetPath().c_str());
            break;
        case lldb::ePathTypeLLDBTempSystemDir:
            COMPUTE_LLDB_PATH(ComputeTempFileDirectory, m_lldb_tmp_dir)
            if (log)
                log->Printf("HostInfoBase::GetLLDBPath(ePathTypeLLDBTempSystemDir) => '%s'", m_lldb_tmp_dir.GetPath().c_str());
            break;
    }

    if (!result)
        return false;
    file_spec = *result;
    return true;
}

bool
HostInfoBase::ComputeSharedLibraryDirectory(FileSpec &file_spec)
{
    // To get paths related to LLDB we get the path to the executable that
    // contains this function. On MacOSX this will be "LLDB.framework/.../LLDB",
    // on linux this is assumed to be the "lldb" main executable. If LLDB on
    // linux is actually in a shared library (liblldb.so) then this function will
    // need to be modified to "do the right thing".

    FileSpec lldb_file_spec(
        Host::GetModuleFileSpecForHostAddress(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(HostInfoBase::GetLLDBPath))));

    // Remove the filename so that this FileSpec only represents the directory.
    file_spec.SetFile(lldb_file_spec.GetDirectory().AsCString(), true);

    return (bool)file_spec.GetDirectory();
}

bool
HostInfoBase::ComputeSupportExeDirectory(FileSpec &file_spec)
{
    return GetLLDBPath(lldb::ePathTypeLLDBShlibDir, file_spec);
}

bool
HostInfoBase::ComputeTempFileDirectory(FileSpec &file_spec)
{
    const char *tmpdir_cstr = getenv("TMPDIR");
    if (tmpdir_cstr == NULL)
    {
        tmpdir_cstr = getenv("TMP");
        if (tmpdir_cstr == NULL)
            tmpdir_cstr = getenv("TEMP");
    }
    if (!tmpdir_cstr)
        return false;

    StreamString pid_tmpdir;
    pid_tmpdir.Printf("%s/lldb", tmpdir_cstr);
    if (!FileSystem::MakeDirectory(pid_tmpdir.GetString().c_str(), eFilePermissionsDirectoryDefault).Success())
        return false;

    pid_tmpdir.Printf("/%" PRIu64, Host::GetCurrentProcessID());
    if (!FileSystem::MakeDirectory(pid_tmpdir.GetString().c_str(), eFilePermissionsDirectoryDefault).Success())
        return false;

    // Make an atexit handler to clean up the process specify LLDB temp dir
    // and all of its contents.
    ::atexit(CleanupProcessSpecificLLDBTempDir);
    file_spec.SetFile(pid_tmpdir.GetString().c_str(), false);
    return true;
}

bool
HostInfoBase::ComputeHeaderDirectory(FileSpec &file_spec)
{
    // TODO(zturner): Figure out how to compute the header directory for all platforms.
    return false;
}

bool
HostInfoBase::ComputeSystemPluginsDirectory(FileSpec &file_spec)
{
    // TODO(zturner): Figure out how to compute the system plugins directory for all platforms.
    return false;
}

bool
HostInfoBase::ComputeUserPluginsDirectory(FileSpec &file_spec)
{
    // TODO(zturner): Figure out how to compute the user plugins directory for all platforms.
    return false;
}

void
HostInfoBase::ComputeHostArchitectureSupport(ArchSpec &arch_32, ArchSpec &arch_64)
{
    llvm::Triple triple(llvm::sys::getDefaultTargetTriple());

    arch_32.Clear();
    arch_64.Clear();

    switch (triple.getArch())
    {
        default:
            arch_32.SetTriple(triple);
            break;

        case llvm::Triple::x86_64:
            arch_64.SetTriple(triple);
            arch_32.SetTriple(triple.get32BitArchVariant());
            break;

        case llvm::Triple::mips64:
        case llvm::Triple::sparcv9:
        case llvm::Triple::ppc64:
            arch_64.SetTriple(triple);
            break;
    }
}
