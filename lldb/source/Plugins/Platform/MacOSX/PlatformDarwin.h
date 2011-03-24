//===-- PlatformDarwin.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformDarwin_h_
#define liblldb_PlatformDarwin_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

namespace lldb_private {

    class PlatformDarwin : public Platform
    {
    public:
        PlatformDarwin (bool is_host);

        virtual
        ~PlatformDarwin();
        
        //------------------------------------------------------------
        // lldb_private::Platform functions
        //------------------------------------------------------------
        virtual Error
        ResolveExecutable (const FileSpec &exe_file,
                           const ArchSpec &arch,
                           lldb::ModuleSP &module_sp);

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

        virtual bool
        IsConnected () const;

        virtual Error
        ConnectRemote (Args& args);

        virtual Error
        DisconnectRemote ();

        virtual const char *
        GetRemoteHostname ();


    protected:
        lldb::PlatformSP m_remote_platform_sp; // Allow multiple ways to connect to a remote darwin OS

    private:
        DISALLOW_COPY_AND_ASSIGN (PlatformDarwin);

    };
} // namespace lldb_private

#endif  // liblldb_PlatformDarwin_h_
