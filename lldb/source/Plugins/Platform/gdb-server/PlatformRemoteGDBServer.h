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

namespace lldb_private {

    class PlatformRemoteGDBServer : public Platform
    {
    public:

        static void
        Initialize ();

        static void
        Terminate ();
        
        static Platform* 
        CreateInstance ();

        static const char *
        GetShortPluginNameStatic();

        static const char *
        GetDescriptionStatic();
    

        PlatformRemoteGDBServer ();

        virtual
        ~PlatformRemoteGDBServer();

        //------------------------------------------------------------
        // lldb_private::PluginInterface functions
        //------------------------------------------------------------
        virtual const char *
        GetPluginName()
        {
            return "PlatformRemoteGDBServer";
        }
        
        virtual const char *
        GetShortPluginName()
        {
            return GetShortPluginNameStatic();
        }
        
        virtual uint32_t
        GetPluginVersion()
        {
            return 1;
        }
        

        //------------------------------------------------------------
        // lldb_private::Platform functions
        //------------------------------------------------------------
        virtual Error
        ResolveExecutable (const FileSpec &exe_file,
                           const ArchSpec &arch,
                           lldb::ModuleSP &module_sp);

        virtual const char *
        GetDescription ();

        virtual Error
        GetFile (const FileSpec &platform_file, 
                 const UUID *uuid_ptr,
                 FileSpec &local_file);

        virtual uint32_t
        FindProcessesByName (const char *name_match, 
                             NameMatchType name_match_type,
                             ProcessInfoList &process_infos);

        virtual bool
        GetProcessInfo (lldb::pid_t pid, ProcessInfo &proc_info);

        virtual bool
        GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch);

        virtual size_t
        GetSoftwareBreakpointTrapOpcode (Target &target, 
                                         BreakpointSite *bp_site);

        virtual bool
        GetRemoteOSVersion ();

        virtual bool
        GetRemoteOSBuildString (std::string &s);
        
        virtual bool
        GetRemoteOSKernelDescription (std::string &s);

        // Remote Platform subclasses need to override this function
        virtual ArchSpec
        GetRemoteSystemArchitecture ();

        // Remote subclasses should override this and return a valid instance
        // name if connected.
        virtual const char *
        GetRemoteHostname ();

        virtual bool
        IsConnected () const;

        virtual Error
        ConnectRemote (Args& args);

        virtual Error
        DisconnectRemote ();

    protected:
        GDBRemoteCommunicationClient m_gdb_client;
        std::string m_platform_description; // After we connect we can get a more complete description of what we are connected to

    private:
        DISALLOW_COPY_AND_ASSIGN (PlatformRemoteGDBServer);

    };
} // namespace lldb_private

#endif  // liblldb_PlatformRemoteGDBServer_h_
