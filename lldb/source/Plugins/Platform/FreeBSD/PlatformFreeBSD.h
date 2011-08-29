//===-- PlatformFreeBSD.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformFreeBSD_h_
#define liblldb_PlatformFreeBSD_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

namespace lldb_private {

    class PlatformFreeBSD : public Platform
    {
    public:

        static void
        Initialize ();

        static void
        Terminate ();
        
        PlatformFreeBSD (bool is_host);

        virtual
        ~PlatformFreeBSD();

        //------------------------------------------------------------
        // lldb_private::PluginInterface functions
        //------------------------------------------------------------
        static Platform *
        CreateInstance ();

        static const char *
        GetPluginNameStatic();

        static const char *
        GetShortPluginNameStatic(bool is_host);

        static const char *
        GetDescriptionStatic(bool is_host);

        virtual const char *
        GetPluginName()
        {
            return GetPluginNameStatic();
        }
        
        virtual const char *
        GetShortPluginName()
        {
            return GetShortPluginNameStatic (IsHost());
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
            return GetDescriptionStatic(IsHost());
        }

        virtual void
        GetStatus (Stream &strm);

        virtual Error
        GetFile (const FileSpec &platform_file,
                 const UUID* uuid, FileSpec &local_file);

        virtual bool
        GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &proc_info);

        virtual bool
        GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch);

        virtual size_t
        GetSoftwareBreakpointTrapOpcode (Target &target, 
                                         BreakpointSite *bp_site);

        virtual lldb::ProcessSP
        Attach(lldb::pid_t pid, Debugger &debugger, Target *target,
               Listener &listener, Error &error);

    protected:
        
        
    private:
        DISALLOW_COPY_AND_ASSIGN (PlatformFreeBSD);
    };
} // namespace lldb_private

#endif  // liblldb_PlatformFreeBSD_h_
