//===-- PlatformLinux.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformLinux_h_
#define liblldb_PlatformLinux_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

namespace lldb_private {

    class PlatformLinux : public Platform
    {
    public:

        static void
        Initialize ();

        static void
        Terminate ();
        
        PlatformLinux ();

        virtual
        ~PlatformLinux();

        //------------------------------------------------------------
        // lldb_private::PluginInterface functions
        //------------------------------------------------------------
        static Platform *
        CreateInstance ();

        static const char *
        GetPluginNameStatic();

        static const char *
        GetPluginDescriptionStatic();

        virtual const char *
        GetPluginName()
        {
            return GetPluginNameStatic();
        }
        
        virtual const char *
        GetShortPluginName()
        {
            return "PlatformLinux";
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
            return GetPluginDescriptionStatic();
        }

        virtual void
        GetStatus (Stream &strm);

        virtual Error
        GetFile (const FileSpec &platform_file,
                 const UUID* uuid, FileSpec &local_file);

        virtual uint32_t
        FindProcesseses (const ProcessInfoMatch &match_info,
                         ProcessInfoList &process_infos);

        virtual bool
        GetProcessInfo (lldb::pid_t pid, ProcessInfo &proc_info);

        virtual bool
        GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch);

        virtual size_t
        GetSoftwareBreakpointTrapOpcode (Target &target, 
                                         BreakpointSite *bp_site);

    protected:
        
        
    private:
        DISALLOW_COPY_AND_ASSIGN (PlatformLinux);
    };
} // namespace lldb_private

#endif  // liblldb_PlatformLinux_h_
