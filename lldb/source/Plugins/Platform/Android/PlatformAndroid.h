//===-- PlatformAndroid.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformAndroid_h_
#define liblldb_PlatformAndroid_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "Plugins/Platform/Linux/PlatformLinux.h"

namespace lldb_private {

    class PlatformAndroid : public PlatformLinux
    {
    public:
        static void
        Initialize ();

        static void
        Terminate ();

        PlatformAndroid (bool is_host);

        virtual
        ~PlatformAndroid();

        //------------------------------------------------------------
        // lldb_private::PluginInterface functions
        //------------------------------------------------------------
        static lldb::PlatformSP
        CreateInstance (bool force, const lldb_private::ArchSpec *arch);

        static lldb_private::ConstString
        GetPluginNameStatic (bool is_host);

        static const char *
        GetPluginDescriptionStatic (bool is_host);

        lldb_private::ConstString
        GetPluginName() override;
        
        uint32_t
        GetPluginVersion() override
        {
            return 1;
        }

        //------------------------------------------------------------
        // lldb_private::Platform functions
        //------------------------------------------------------------

        lldb_private::Error
        ConnectRemote (lldb_private::Args& args) override;

    private:
        DISALLOW_COPY_AND_ASSIGN (PlatformAndroid);
    };
} // namespace lldb_private

#endif  // liblldb_PlatformAndroid_h_
