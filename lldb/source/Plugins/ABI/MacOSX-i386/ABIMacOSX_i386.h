//===-- ABIMacOSX_i386.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABIMacOSX_i386_h_
#define liblldb_ABIMacOSX_i386_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ABI.h"
#include "lldb/Core/Value.h"

namespace lldb_private {
    
    class ABIMacOSX_i386 :
    public lldb_private::ABI
    {
    public:
        ~ABIMacOSX_i386() { }
        
        virtual size_t 
        GetRedZoneSize () const;
        
        virtual bool
        PrepareTrivialCall (Thread &thread, 
                            lldb::addr_t sp,
                            lldb::addr_t functionAddress,
                            lldb::addr_t returnAddress, 
                            lldb::addr_t arg,
                            lldb::addr_t *this_arg,
                            lldb::addr_t *cmd_arg) const;
        
        virtual bool
        PrepareNormalCall (Thread &thread,
                           lldb::addr_t sp,
                           lldb::addr_t functionAddress,
                           lldb::addr_t returnAddress,
                           ValueList &args) const;
        
        virtual bool
        GetArgumentValues (Thread &thread,
                           ValueList &values) const;
        
        virtual bool
        GetReturnValue (Thread &thread,
                        Value &value) const;
        
        //------------------------------------------------------------------
        // Static Functions
        //------------------------------------------------------------------
        static void
        Initialize();
        
        static void
        Terminate();
        
        static lldb_private::ABI *
        CreateInstance (const ConstString &triple);
        
        //------------------------------------------------------------------
        // PluginInterface protocol
        //------------------------------------------------------------------
        virtual const char *
        GetPluginName();
        
        virtual const char *
        GetShortPluginName();
        
        virtual uint32_t
        GetPluginVersion();
        
        virtual void
        GetPluginCommandHelp (const char *command, lldb_private::Stream *strm);
        
        virtual lldb_private::Error
        ExecutePluginCommand (lldb_private::Args &command, lldb_private::Stream *strm);
        
        virtual lldb_private::Log *
        EnablePluginLogging (lldb_private::Stream *strm, lldb_private::Args &command);
    protected:
    private:
        ABIMacOSX_i386() : lldb_private::ABI() { } // Call CreateInstance instead.
    };
    
} // namespace lldb_private

#endif  // liblldb_ABI_h_
