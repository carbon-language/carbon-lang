//===-- PlatformRemoteiOS.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformRemoteiOS_h_
#define liblldb_PlatformRemoteiOS_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

namespace lldb_private {

    class PlatformRemoteiOS : public Platform
    {
    public:

        static Platform* 
        CreateInstance ();

        static void
        Initialize ();

        static void
        Terminate ();
        
        PlatformRemoteiOS ();

        virtual
        ~PlatformRemoteiOS();

        //------------------------------------------------------------
        // lldb_private::PluginInterface functions
        //------------------------------------------------------------
        
        static const char *
        GetPluginNameStatic ();

        static const char *
        GetShortPluginNameStatic();

        static const char *
        GetDescriptionStatic();

        virtual const char *
        GetPluginName()
        {
            return GetPluginNameStatic();
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
        GetDescription ()
        {
            return GetDescriptionStatic();
        }

        virtual void
        GetStatus (Stream &strm);

        virtual Error
        GetFile (const FileSpec &platform_file, FileSpec &local_file);

        virtual uint32_t
        FindProcessesByName (const char *name_match, 
                             lldb::NameMatchType name_match_type,
                             ProcessInfoList &process_infos);

        virtual bool
        GetProcessInfo (lldb::pid_t pid, ProcessInfo &proc_info);

        virtual bool
        GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch);

        virtual size_t
        GetSoftwareBreakpointTrapOpcode (Target &target, 
                                         BreakpointSite *bp_site);

    protected:
        std::string m_device_support_directory;
        std::string m_device_support_directory_for_os_version;
        std::string m_build_update;
        //std::vector<FileSpec> m_device_support_os_dirs;
        
        const char *
        GetDeviceSupportDirectory();

        const char *
        GetDeviceSupportDirectoryForOSVersion();

    private:
        DISALLOW_COPY_AND_ASSIGN (PlatformRemoteiOS);

    };
} // namespace lldb_private

#endif  // liblldb_Platform_h_
