//===-- PlatformRemoteGDBServer.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformRemoteGDBServer_h_
#define liblldb_PlatformRemoteGDBServer_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"
#include "../../Process/gdb-remote/GDBRemoteCommunicationClient.h"

class PlatformRemoteGDBServer : public lldb_private::Platform
{
public:

    static void
    Initialize ();

    static void
    Terminate ();
    
    static lldb::PlatformSP
    CreateInstance (bool force, const lldb_private::ArchSpec *arch);

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetDescriptionStatic();


    PlatformRemoteGDBServer ();

    virtual
    ~PlatformRemoteGDBServer();

    //------------------------------------------------------------
    // lldb_private::PluginInterface functions
    //------------------------------------------------------------
    lldb_private::ConstString
    GetPluginName() override
    {
        return GetPluginNameStatic();
    }
    
    uint32_t
    GetPluginVersion() override
    {
        return 1;
    }
    

    //------------------------------------------------------------
    // lldb_private::Platform functions
    //------------------------------------------------------------
    lldb_private::Error
    ResolveExecutable (const lldb_private::ModuleSpec &module_spec,
                       lldb::ModuleSP &module_sp,
                       const lldb_private::FileSpecList *module_search_paths_ptr) override;

    bool
    GetModuleSpec (const lldb_private::FileSpec& module_file_spec,
                   const lldb_private::ArchSpec& arch,
                   lldb_private::ModuleSpec &module_spec) override;

    const char *
    GetDescription () override;

    lldb_private::Error
    GetFileWithUUID (const lldb_private::FileSpec &platform_file, 
                     const lldb_private::UUID *uuid_ptr,
                     lldb_private::FileSpec &local_file) override;

    bool
    GetProcessInfo (lldb::pid_t pid, 
                    lldb_private::ProcessInstanceInfo &proc_info) override;
    
    uint32_t
    FindProcesses (const lldb_private::ProcessInstanceInfoMatch &match_info,
                   lldb_private::ProcessInstanceInfoList &process_infos) override;

    lldb_private::Error
    LaunchProcess (lldb_private::ProcessLaunchInfo &launch_info) override;

    lldb_private::Error
    KillProcess (const lldb::pid_t pid) override;

    lldb::ProcessSP
    DebugProcess (lldb_private::ProcessLaunchInfo &launch_info,
                  lldb_private::Debugger &debugger,
                  lldb_private::Target *target,       // Can be NULL, if NULL create a new target, else use existing one
                  lldb_private::Error &error) override;

    lldb::ProcessSP
    Attach (lldb_private::ProcessAttachInfo &attach_info,
            lldb_private::Debugger &debugger,
            lldb_private::Target *target,       // Can be NULL, if NULL create a new target, else use existing one
            lldb_private::Error &error) override;

    bool
    GetSupportedArchitectureAtIndex (uint32_t idx, lldb_private::ArchSpec &arch) override;

    size_t
    GetSoftwareBreakpointTrapOpcode (lldb_private::Target &target, 
                                     lldb_private::BreakpointSite *bp_site) override;

    bool
    GetRemoteOSVersion () override;

    bool
    GetRemoteOSBuildString (std::string &s) override;
    
    bool
    GetRemoteOSKernelDescription (std::string &s) override;

    // Remote Platform subclasses need to override this function
    lldb_private::ArchSpec
    GetRemoteSystemArchitecture () override;

    lldb_private::ConstString
    GetRemoteWorkingDirectory() override;
    
    bool
    SetRemoteWorkingDirectory(const lldb_private::ConstString &path) override;

    // Remote subclasses should override this and return a valid instance
    // name if connected.
    const char *
    GetHostname () override;

    const char *
    GetUserName (uint32_t uid) override;
    
    const char *
    GetGroupName (uint32_t gid) override;

    bool
    IsConnected () const override;

    lldb_private::Error
    ConnectRemote (lldb_private::Args& args) override;

    lldb_private::Error
    DisconnectRemote () override;
    
    lldb_private::Error
    MakeDirectory (const char *path, uint32_t file_permissions) override;
    
    lldb_private::Error
    GetFilePermissions (const char *path, uint32_t &file_permissions) override;
    
    lldb_private::Error
    SetFilePermissions (const char *path, uint32_t file_permissions) override;
    

    lldb::user_id_t
    OpenFile (const lldb_private::FileSpec& file_spec,
              uint32_t flags,
              uint32_t mode,
              lldb_private::Error &error) override;
    
    bool
    CloseFile (lldb::user_id_t fd,
               lldb_private::Error &error) override;
    
    uint64_t
    ReadFile (lldb::user_id_t fd,
              uint64_t offset,
              void *data_ptr,
              uint64_t len,
              lldb_private::Error &error) override;
    
    uint64_t
    WriteFile (lldb::user_id_t fd,
               uint64_t offset,
               const void* data,
               uint64_t len,
               lldb_private::Error &error) override;

    lldb::user_id_t
    GetFileSize (const lldb_private::FileSpec& file_spec) override;

    lldb_private::Error
    PutFile (const lldb_private::FileSpec& source,
             const lldb_private::FileSpec& destination,
             uint32_t uid = UINT32_MAX,
             uint32_t gid = UINT32_MAX) override;
    
    lldb_private::Error
    CreateSymlink (const char *src, const char *dst) override;

    bool
    GetFileExists (const lldb_private::FileSpec& file_spec) override;

    lldb_private::Error
    Unlink (const char *path) override;

    lldb_private::Error
    RunShellCommand (const char *command,            // Shouldn't be NULL
                     const char *working_dir,        // Pass NULL to use the current working directory
                     int *status_ptr,                // Pass NULL if you don't want the process exit status
                     int *signo_ptr,                 // Pass NULL if you don't want the signal that caused the process to exit
                     std::string *command_output,    // Pass NULL if you don't want the command output
                     uint32_t timeout_sec) override; // Timeout in seconds to wait for shell program to finish

    void
    CalculateTrapHandlerSymbolNames () override;

protected:
    GDBRemoteCommunicationClient m_gdb_client;
    std::string m_platform_description; // After we connect we can get a more complete description of what we are connected to
    std::string m_platform_hostname;

    // Launch the lldb-gdbserver on the remote host and return the port it is listening on or 0 on
    // failure. Subclasses should override this method if they want to do extra actions before or
    // after launching the lldb-gdbserver.
    virtual uint16_t
    LaunchGDBserverAndGetPort (lldb::pid_t &pid);

    virtual bool
    KillSpawnedProcess (lldb::pid_t pid);

private:
    DISALLOW_COPY_AND_ASSIGN (PlatformRemoteGDBServer);

};

#endif  // liblldb_PlatformRemoteGDBServer_h_
