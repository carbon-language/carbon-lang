//===-- PlatformMacOSX.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformMacOSX_h_
#define liblldb_PlatformMacOSX_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

namespace lldb_private {

    class PlatformMacOSX : public Platform
    {
    public:

        static void
        Initialize ();

        static void
        Terminate ();
        
        PlatformMacOSX ();

        virtual
        ~PlatformMacOSX();

        //------------------------------------------------------------
        // lldb_private::PluginInterface functions
        //------------------------------------------------------------
        virtual const char *
        GetPluginName()
        {
            return "PlatformMacOSX";
        }
        
        virtual const char *
        GetShortPluginName()
        {
            return "local-macosx";
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
            return "The native host platform on MacOSX.";
        }

        virtual void
        GetStatus (Stream &strm);

        virtual Error
        GetFile (const FileSpec &platform_file, 
                 const UUID *uuid_ptr,
                 FileSpec &local_file);

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

    private:
        DISALLOW_COPY_AND_ASSIGN (PlatformMacOSX);

    };
} // namespace lldb_private

#endif  // liblldb_PlatformMacOSX_h_
