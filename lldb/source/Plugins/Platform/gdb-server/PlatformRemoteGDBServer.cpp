//===-- PlatformRemoteGDBServer.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "PlatformRemoteGDBServer.h"
#include "lldb/Host/Config.h"

// C Includes
#ifndef LLDB_DISABLE_POSIX
#include <sys/sysctl.h>
#endif

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

static bool g_initialized = false;

void
PlatformRemoteGDBServer::Initialize ()
{
    if (g_initialized == false)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin (PlatformRemoteGDBServer::GetPluginNameStatic(),
                                       PlatformRemoteGDBServer::GetDescriptionStatic(),
                                       PlatformRemoteGDBServer::CreateInstance);
    }
}

void
PlatformRemoteGDBServer::Terminate ()
{
    if (g_initialized)
    {
        g_initialized = false;
        PluginManager::UnregisterPlugin (PlatformRemoteGDBServer::CreateInstance);
    }
}

Platform* 
PlatformRemoteGDBServer::CreateInstance (bool force, const lldb_private::ArchSpec *arch)
{
    bool create = force;
    if (!create)
    {
        create = !arch->TripleVendorWasSpecified() && !arch->TripleOSWasSpecified();
    }
    if (create)
        return new PlatformRemoteGDBServer ();
    return NULL;
}


lldb_private::ConstString
PlatformRemoteGDBServer::GetPluginNameStatic()
{
    static ConstString g_name("remote-gdb-server");
    return g_name;
}

const char *
PlatformRemoteGDBServer::GetDescriptionStatic()
{
    return "A platform that uses the GDB remote protocol as the communication transport.";
}

const char *
PlatformRemoteGDBServer::GetDescription ()
{
    if (m_platform_description.empty())
    {
        if (IsConnected())
        {
            // Send the get description packet
        }
    }
    
    if (!m_platform_description.empty())
        return m_platform_description.c_str();
    return GetDescriptionStatic();
}

Error
PlatformRemoteGDBServer::ResolveExecutable (const FileSpec &exe_file,
                                            const ArchSpec &exe_arch,
                                            lldb::ModuleSP &exe_module_sp,
                                            const FileSpecList *module_search_paths_ptr)
{
    Error error;
    //error.SetErrorString ("PlatformRemoteGDBServer::ResolveExecutable() is unimplemented");
    if (m_gdb_client.GetFileExists(exe_file))
        return error;
    // TODO: get the remote end to somehow resolve this file
    error.SetErrorString("file not found on remote end");
    return error;
}

Error
PlatformRemoteGDBServer::GetFileWithUUID (const FileSpec &platform_file, 
                                          const UUID *uuid_ptr,
                                          FileSpec &local_file)
{
    // Default to the local case
    local_file = platform_file;
    return Error();
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformRemoteGDBServer::PlatformRemoteGDBServer () :
    Platform(false), // This is a remote platform
    m_gdb_client(true)
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformRemoteGDBServer::~PlatformRemoteGDBServer()
{
}

bool
PlatformRemoteGDBServer::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    return false;
}

size_t
PlatformRemoteGDBServer::GetSoftwareBreakpointTrapOpcode (Target &target, BreakpointSite *bp_site)
{
    // This isn't needed if the z/Z packets are supported in the GDB remote
    // server. But we might need a packet to detect this.
    return 0;
}

bool
PlatformRemoteGDBServer::GetRemoteOSVersion ()
{
    uint32_t major, minor, update;
    if (m_gdb_client.GetOSVersion (major, minor, update))
    {
        m_major_os_version = major;
        m_minor_os_version = minor;
        m_update_os_version = update;
        return true;
    }
    return false;
}

bool
PlatformRemoteGDBServer::GetRemoteOSBuildString (std::string &s)
{
    return m_gdb_client.GetOSBuildString (s);
}

bool
PlatformRemoteGDBServer::GetRemoteOSKernelDescription (std::string &s)
{
    return m_gdb_client.GetOSKernelDescription (s);
}

// Remote Platform subclasses need to override this function
ArchSpec
PlatformRemoteGDBServer::GetRemoteSystemArchitecture ()
{
    return m_gdb_client.GetSystemArchitecture();
}

lldb_private::ConstString
PlatformRemoteGDBServer::GetRemoteWorkingDirectory()
{
    if (IsConnected())
    {
        Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
        std::string cwd;
        if (m_gdb_client.GetWorkingDir(cwd))
        {
            ConstString working_dir(cwd.c_str());
            if (log)
                log->Printf("PlatformRemoteGDBServer::GetRemoteWorkingDirectory() -> '%s'", working_dir.GetCString());
            return working_dir;
        }
        else
        {
            return ConstString();
        }
    }
    else
    {
        return Platform::GetRemoteWorkingDirectory();
    }
}

bool
PlatformRemoteGDBServer::SetRemoteWorkingDirectory(const lldb_private::ConstString &path)
{
    if (IsConnected())
    {
        // Clear the working directory it case it doesn't get set correctly. This will
        // for use to re-read it
        Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
        if (log)
            log->Printf("PlatformRemoteGDBServer::SetRemoteWorkingDirectory('%s')", path.GetCString());
        return m_gdb_client.SetWorkingDir(path.GetCString()) == 0;
    }
    else
        return Platform::SetRemoteWorkingDirectory(path);
}

bool
PlatformRemoteGDBServer::IsConnected () const
{
    return m_gdb_client.IsConnected();
}        

Error
PlatformRemoteGDBServer::ConnectRemote (Args& args)
{
    Error error;
    if (IsConnected())
    {
        error.SetErrorStringWithFormat ("the platform is already connected to '%s', execute 'platform disconnect' to close the current connection", 
                                        GetHostname());
    }
    else
    {
        if (args.GetArgumentCount() == 1)
        {
            const char *url = args.GetArgumentAtIndex(0);
            m_gdb_client.SetConnection (new ConnectionFileDescriptor());
            const ConnectionStatus status = m_gdb_client.Connect(url, &error);
            if (status == eConnectionStatusSuccess)
            {
                if (m_gdb_client.HandshakeWithServer(&error))
                {
                    m_gdb_client.GetHostInfo();
                    // If a working directory was set prior to connecting, send it down now
                    if (m_working_dir)
                        m_gdb_client.SetWorkingDir(m_working_dir.GetCString());
                }
                else
                {
                    m_gdb_client.Disconnect();
                    if (error.Success())
                        error.SetErrorString("handshake failed");
                }
            }
        }
        else
        {
            error.SetErrorString ("\"platform connect\" takes a single argument: <connect-url>");
        }
    }

    return error;
}

Error
PlatformRemoteGDBServer::DisconnectRemote ()
{
    Error error;
    m_gdb_client.Disconnect(&error);
    return error;
}

const char *
PlatformRemoteGDBServer::GetHostname ()
{
    m_gdb_client.GetHostname (m_name);
    if (m_name.empty())
        return NULL;
    return m_name.c_str();
}

const char *
PlatformRemoteGDBServer::GetUserName (uint32_t uid)
{
    // Try and get a cache user name first
    const char *cached_user_name = Platform::GetUserName(uid);
    if (cached_user_name)
        return cached_user_name;
    std::string name;
    if (m_gdb_client.GetUserName(uid, name))
        return SetCachedUserName(uid, name.c_str(), name.size());

    SetUserNameNotFound(uid); // Negative cache so we don't keep sending packets
    return NULL;
}

const char *
PlatformRemoteGDBServer::GetGroupName (uint32_t gid)
{
    const char *cached_group_name = Platform::GetGroupName(gid);
    if (cached_group_name)
        return cached_group_name;
    std::string name;
    if (m_gdb_client.GetGroupName(gid, name))
        return SetCachedGroupName(gid, name.c_str(), name.size());

    SetGroupNameNotFound(gid); // Negative cache so we don't keep sending packets
    return NULL;
}

uint32_t
PlatformRemoteGDBServer::FindProcesses (const ProcessInstanceInfoMatch &match_info,
                                        ProcessInstanceInfoList &process_infos)
{
    return m_gdb_client.FindProcesses (match_info, process_infos);
}

bool
PlatformRemoteGDBServer::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    return m_gdb_client.GetProcessInfo (pid, process_info);
}


Error
PlatformRemoteGDBServer::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Error error;
    lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
    
    m_gdb_client.SetSTDIN ("/dev/null");
    m_gdb_client.SetSTDOUT ("/dev/null");
    m_gdb_client.SetSTDERR ("/dev/null");
    m_gdb_client.SetDisableASLR (launch_info.GetFlags().Test (eLaunchFlagDisableASLR));
    
    const char *working_dir = launch_info.GetWorkingDirectory();
    if (working_dir && working_dir[0])
    {
        m_gdb_client.SetWorkingDir (working_dir);
    }
    
    // Send the environment and the program + arguments after we connect
    const char **envp = launch_info.GetEnvironmentEntries().GetConstArgumentVector();

    if (envp)
    {
        const char *env_entry;
        for (int i=0; (env_entry = envp[i]); ++i)
        {
            if (m_gdb_client.SendEnvironmentPacket(env_entry) != 0)
                break;
        }
    }
    
    ArchSpec arch_spec = launch_info.GetArchitecture();
    const char *arch_triple = arch_spec.GetTriple().str().c_str();
    
    m_gdb_client.SendLaunchArchPacket(arch_triple);
    
    const uint32_t old_packet_timeout = m_gdb_client.SetPacketTimeout (5);
    int arg_packet_err = m_gdb_client.SendArgumentsPacket (launch_info);
    m_gdb_client.SetPacketTimeout (old_packet_timeout);
    if (arg_packet_err == 0)
    {
        std::string error_str;
        if (m_gdb_client.GetLaunchSuccess (error_str))
        {
            pid = m_gdb_client.GetCurrentProcessID ();
            if (pid != LLDB_INVALID_PROCESS_ID)
                launch_info.SetProcessID (pid);
        }
        else
        {
            error.SetErrorString (error_str.c_str());
        }
    }
    else
    {
        error.SetErrorStringWithFormat("'A' packet returned an error: %i", arg_packet_err);
    }
    return error;
}

lldb::ProcessSP
PlatformRemoteGDBServer::DebugProcess (lldb_private::ProcessLaunchInfo &launch_info,
                                       lldb_private::Debugger &debugger,
                                       lldb_private::Target *target,       // Can be NULL, if NULL create a new target, else use existing one
                                       lldb_private::Listener &listener,
                                       lldb_private::Error &error)
{
    lldb::ProcessSP process_sp;
    if (IsRemote())
    {
        if (IsConnected())
        {
            lldb::pid_t debugserver_pid = LLDB_INVALID_PROCESS_ID;
            ArchSpec remote_arch = GetRemoteSystemArchitecture();
            llvm::Triple &remote_triple = remote_arch.GetTriple();
            uint16_t port = 0;
            if (remote_triple.getVendor() == llvm::Triple::Apple && remote_triple.getOS() == llvm::Triple::IOS)
            {
                // When remote debugging to iOS, we use a USB mux that always talks
                // to localhost, so we will need the remote debugserver to accept connections
                // only from localhost, no matter what our current hostname is
                port = m_gdb_client.LaunchGDBserverAndGetPort(debugserver_pid, "localhost");
            }
            else
            {
                // All other hosts should use their actual hostname
                port = m_gdb_client.LaunchGDBserverAndGetPort(debugserver_pid, NULL);
            }
            
            if (port == 0)
            {
                error.SetErrorStringWithFormat ("unable to launch a GDB server on '%s'", GetHostname ());
            }
            else
            {
                if (target == NULL)
                {
                    TargetSP new_target_sp;
                    
                    error = debugger.GetTargetList().CreateTarget (debugger,
                                                                   NULL,
                                                                   NULL,
                                                                   false,
                                                                   NULL,
                                                                   new_target_sp);
                    target = new_target_sp.get();
                }
                else
                    error.Clear();
                
                if (target && error.Success())
                {
                    debugger.GetTargetList().SetSelectedTarget(target);
                    
                    // The darwin always currently uses the GDB remote debugger plug-in
                    // so even when debugging locally we are debugging remotely!
                    process_sp = target->CreateProcess (listener, "gdb-remote", NULL);
                    
                    if (process_sp)
                    {
                        char connect_url[256];
                        const char *override_hostname = getenv("LLDB_PLATFORM_REMOTE_GDB_SERVER_HOSTNAME");
                        const char *port_offset_c_str = getenv("LLDB_PLATFORM_REMOTE_GDB_SERVER_PORT_OFFSET");
                        int port_offset = port_offset_c_str ? ::atoi(port_offset_c_str) : 0;
                        const int connect_url_len = ::snprintf (connect_url,
                                                                sizeof(connect_url),
                                                                "connect://%s:%u",
                                                                override_hostname ? override_hostname : GetHostname (),
                                                                port + port_offset);
                        assert (connect_url_len < (int)sizeof(connect_url));
                        error = process_sp->ConnectRemote (NULL, connect_url);
                        if (error.Success())
                            error = process_sp->Launch(launch_info);
                        else if (debugserver_pid != LLDB_INVALID_PROCESS_ID)
                            m_gdb_client.KillSpawnedProcess(debugserver_pid);
                    }
                }
            }
        }
        else
        {
            error.SetErrorString("not connected to remote gdb server");
        }
    }
    return process_sp;
    
}

lldb::ProcessSP
PlatformRemoteGDBServer::Attach (lldb_private::ProcessAttachInfo &attach_info,
                                 Debugger &debugger,
                                 Target *target,       // Can be NULL, if NULL create a new target, else use existing one
                                 Listener &listener, 
                                 Error &error)
{
    lldb::ProcessSP process_sp;
    if (IsRemote())
    {
        if (IsConnected())
        {
            lldb::pid_t debugserver_pid = LLDB_INVALID_PROCESS_ID;
            ArchSpec remote_arch = GetRemoteSystemArchitecture();
            llvm::Triple &remote_triple = remote_arch.GetTriple();
            uint16_t port = 0;
            if (remote_triple.getVendor() == llvm::Triple::Apple && remote_triple.getOS() == llvm::Triple::IOS)
            {
                // When remote debugging to iOS, we use a USB mux that always talks
                // to localhost, so we will need the remote debugserver to accept connections
                // only from localhost, no matter what our current hostname is
                port = m_gdb_client.LaunchGDBserverAndGetPort(debugserver_pid, "localhost");
            }
            else
            {
                // All other hosts should use their actual hostname
                port = m_gdb_client.LaunchGDBserverAndGetPort(debugserver_pid, NULL);
            }
            
            if (port == 0)
            {
                error.SetErrorStringWithFormat ("unable to launch a GDB server on '%s'", GetHostname ());
            }
            else
            {
                if (target == NULL)
                {
                    TargetSP new_target_sp;
                    
                    error = debugger.GetTargetList().CreateTarget (debugger,
                                                                   NULL,
                                                                   NULL, 
                                                                   false,
                                                                   NULL,
                                                                   new_target_sp);
                    target = new_target_sp.get();
                }
                else
                    error.Clear();
                
                if (target && error.Success())
                {
                    debugger.GetTargetList().SetSelectedTarget(target);
                    
                    // The darwin always currently uses the GDB remote debugger plug-in
                    // so even when debugging locally we are debugging remotely!
                    process_sp = target->CreateProcess (listener, "gdb-remote", NULL);
                    
                    if (process_sp)
                    {
                        char connect_url[256];
                        const char *override_hostname = getenv("LLDB_PLATFORM_REMOTE_GDB_SERVER_HOSTNAME");
                        const char *port_offset_c_str = getenv("LLDB_PLATFORM_REMOTE_GDB_SERVER_PORT_OFFSET");
                        int port_offset = port_offset_c_str ? ::atoi(port_offset_c_str) : 0;
                        const int connect_url_len = ::snprintf (connect_url, 
                                                                sizeof(connect_url), 
                                                                "connect://%s:%u", 
                                                                override_hostname ? override_hostname : GetHostname (), 
                                                                port + port_offset);
                        assert (connect_url_len < (int)sizeof(connect_url));
                        error = process_sp->ConnectRemote (NULL, connect_url);
                        if (error.Success())
                            error = process_sp->Attach(attach_info);
                        else if (debugserver_pid != LLDB_INVALID_PROCESS_ID)
                        {
                            m_gdb_client.KillSpawnedProcess(debugserver_pid);
                        }
                    }
                }
            }
        }
        else
        {
            error.SetErrorString("not connected to remote gdb server");
        }
    }
    return process_sp;
}

Error
PlatformRemoteGDBServer::MakeDirectory (const char *path, uint32_t mode)
{
    Error error = m_gdb_client.MakeDirectory(path,mode);
    Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
    if (log)
        log->Printf ("PlatformRemoteGDBServer::MakeDirectory(path='%s', mode=%o) error = %u (%s)", path, mode, error.GetError(), error.AsCString());
    return error;
}


Error
PlatformRemoteGDBServer::GetFilePermissions (const char *path, uint32_t &file_permissions)
{
    Error error = m_gdb_client.GetFilePermissions(path, file_permissions);
    Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
    if (log)
        log->Printf ("PlatformRemoteGDBServer::GetFilePermissions(path='%s', file_permissions=%o) error = %u (%s)", path, file_permissions, error.GetError(), error.AsCString());
    return error;
}

Error
PlatformRemoteGDBServer::SetFilePermissions (const char *path, uint32_t file_permissions)
{
    Error error = m_gdb_client.SetFilePermissions(path, file_permissions);
    Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
    if (log)
        log->Printf ("PlatformRemoteGDBServer::SetFilePermissions(path='%s', file_permissions=%o) error = %u (%s)", path, file_permissions, error.GetError(), error.AsCString());
    return error;
}


lldb::user_id_t
PlatformRemoteGDBServer::OpenFile (const lldb_private::FileSpec& file_spec,
                                   uint32_t flags,
                                   uint32_t mode,
                                   Error &error)
{
    return m_gdb_client.OpenFile (file_spec, flags, mode, error);
}

bool
PlatformRemoteGDBServer::CloseFile (lldb::user_id_t fd, Error &error)
{
    return m_gdb_client.CloseFile (fd, error);
}

lldb::user_id_t
PlatformRemoteGDBServer::GetFileSize (const lldb_private::FileSpec& file_spec)
{
    return m_gdb_client.GetFileSize(file_spec);
}

uint64_t
PlatformRemoteGDBServer::ReadFile (lldb::user_id_t fd,
                                   uint64_t offset,
                                   void *dst,
                                   uint64_t dst_len,
                                   Error &error)
{
    return m_gdb_client.ReadFile (fd, offset, dst, dst_len, error);
}

uint64_t
PlatformRemoteGDBServer::WriteFile (lldb::user_id_t fd,
                                    uint64_t offset,
                                    const void* src,
                                    uint64_t src_len,
                                    Error &error)
{
    return m_gdb_client.WriteFile (fd, offset, src, src_len, error);
}

lldb_private::Error
PlatformRemoteGDBServer::PutFile (const lldb_private::FileSpec& source,
         const lldb_private::FileSpec& destination,
         uint32_t uid,
         uint32_t gid)
{
    return Platform::PutFile(source,destination,uid,gid);
}

Error
PlatformRemoteGDBServer::CreateSymlink (const char *src,    // The name of the link is in src
                                        const char *dst)    // The symlink points to dst
{
    Error error = m_gdb_client.CreateSymlink (src, dst);
    Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
    if (log)
        log->Printf ("PlatformRemoteGDBServer::CreateSymlink(src='%s', dst='%s') error = %u (%s)", src, dst, error.GetError(), error.AsCString());
    return error;
}

Error
PlatformRemoteGDBServer::Unlink (const char *path)
{
    Error error = m_gdb_client.Unlink (path);
    Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
    if (log)
        log->Printf ("PlatformRemoteGDBServer::Unlink(path='%s') error = %u (%s)", path, error.GetError(), error.AsCString());
    return error;
}

bool
PlatformRemoteGDBServer::GetFileExists (const lldb_private::FileSpec& file_spec)
{
    return m_gdb_client.GetFileExists (file_spec);
}

lldb_private::Error
PlatformRemoteGDBServer::RunShellCommand (const char *command,           // Shouldn't be NULL
                                          const char *working_dir,       // Pass NULL to use the current working directory
                                          int *status_ptr,               // Pass NULL if you don't want the process exit status
                                          int *signo_ptr,                // Pass NULL if you don't want the signal that caused the process to exit
                                          std::string *command_output,   // Pass NULL if you don't want the command output
                                          uint32_t timeout_sec)          // Timeout in seconds to wait for shell program to finish
{
    return m_gdb_client.RunShellCommand (command, working_dir, status_ptr, signo_ptr, command_output, timeout_sec);
}

void
PlatformRemoteGDBServer::CalculateTrapHandlerSymbolNames ()
{   
    m_trap_handlers.push_back (ConstString ("_sigtramp"));
}   
