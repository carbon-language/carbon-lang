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
#include "PlatformDarwin.h"

namespace lldb_private {

    class PlatformMacOSX : public PlatformDarwin
    {
    public:

        //------------------------------------------------------------
        // Class functions
        //------------------------------------------------------------
        static Platform* 
        CreateInstance ();

        static void
        Initialize ();

        static void
        Terminate ();
        
        static const char *
        GetPluginNameStatic ();

        static const char *
        GetShortPluginNameStatic(bool is_host);

        static const char *
        GetDescriptionStatic(bool is_host);
        
        //------------------------------------------------------------
        // Class Methods
        //------------------------------------------------------------
        PlatformMacOSX (bool is_host);

        virtual
        ~PlatformMacOSX();

        //------------------------------------------------------------
        // lldb_private::PluginInterface functions
        //------------------------------------------------------------
        virtual const char *
        GetPluginName()
        {
            return GetPluginNameStatic ();
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
        
        virtual const char *
        GetDescription ()
        {
            return GetDescriptionStatic (IsHost());
        }

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

    private:
        DISALLOW_COPY_AND_ASSIGN (PlatformMacOSX);

    };
} // namespace lldb_private

#endif  // liblldb_PlatformMacOSX_h_
