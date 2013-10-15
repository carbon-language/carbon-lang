//===-- PlatformWindows.h --------------------------------------/*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformWindows_h_
#define liblldb_PlatformWindows_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

namespace lldb_private
{

class PlatformWindows : public Platform
{
public:

    static void
    Initialize(void);

    static void
    Terminate(void);

    PlatformWindows(bool is_host);

    virtual
    ~PlatformWindows(void);

    //------------------------------------------------------------
    // lldb_private::PluginInterface functions
    //------------------------------------------------------------
    static lldb_private::Platform*
    CreateInstance (bool force, const lldb_private::ArchSpec *arch);


    static lldb_private::ConstString
    GetPluginNameStatic(bool is_host);

    static const char *
    GetPluginDescriptionStatic(bool is_host);

    virtual lldb_private::ConstString
    GetPluginName(void);

    virtual uint32_t
    GetPluginVersion(void)
    {
        return 1;
    }

    //------------------------------------------------------------
    // lldb_private::Platform functions
    //------------------------------------------------------------
    virtual Error
    ResolveExecutable(const FileSpec &exe_file,
                      const ArchSpec &arch,
                      lldb::ModuleSP &module_sp,
                      const FileSpecList *module_search_paths_ptr);

    virtual const char *
    GetDescription(void)
    {
        return GetPluginDescriptionStatic(IsHost());
    }

    virtual size_t
    GetSoftwareBreakpointTrapOpcode(lldb_private::Target &target,
                                    lldb_private::BreakpointSite *bp_site);

    virtual bool
    GetRemoteOSVersion(void);

    virtual bool
    GetRemoteOSBuildString(std::string &s);

    virtual bool
    GetRemoteOSKernelDescription(std::string &s);

    // Remote Platform subclasses need to override this function
    virtual lldb_private::ArchSpec
    GetRemoteSystemArchitecture(void);

    virtual bool
    IsConnected(void) const;

    virtual lldb_private::Error
    ConnectRemote(lldb_private::Args& args);

    virtual lldb_private::Error
    DisconnectRemote( void );

    virtual const char *
    GetHostname( void );

    virtual const char *
    GetUserName(uint32_t uid);

    virtual const char *
    GetGroupName(uint32_t gid);

    virtual bool
    GetProcessInfo(lldb::pid_t pid,
                   lldb_private::ProcessInstanceInfo &proc_info);

    virtual uint32_t
    FindProcesses(const lldb_private::ProcessInstanceInfoMatch &match_info,
                  lldb_private::ProcessInstanceInfoList &process_infos);

    virtual lldb_private::Error
    LaunchProcess(lldb_private::ProcessLaunchInfo &launch_info);

    virtual lldb::ProcessSP
    Attach(lldb_private::ProcessAttachInfo &attach_info,
           lldb_private::Debugger &debugger,
           lldb_private::Target *target,
           lldb_private::Listener &listener,
           lldb_private::Error &error);

    virtual lldb_private::Error
    GetFile(const lldb_private::FileSpec &platform_file,
            const lldb_private::UUID* uuid, lldb_private::FileSpec &local_file);

    lldb_private::Error
    GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                    lldb::ModuleSP &module_sp,
                    const lldb_private::FileSpecList *module_search_paths_ptr,
                    lldb::ModuleSP *old_module_sp_ptr,
                    bool *did_create_ptr);

    virtual bool
    GetSupportedArchitectureAtIndex(uint32_t idx, lldb_private::ArchSpec &arch);

    virtual void
    GetStatus(lldb_private::Stream &strm);

    // Local debugging not yet supported
    virtual bool
    CanDebugProcess(void)
    {
        return false;
    }

protected:
    lldb::PlatformSP m_remote_platform_sp;

private:
    DISALLOW_COPY_AND_ASSIGN (PlatformWindows);
};

}

#endif  // liblldb_PlatformWindows_h_
