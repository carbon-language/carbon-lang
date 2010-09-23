//===-- ItaniumABILanguageRuntime.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ItaniumABILanguageRuntime_h_
#define liblldb_ItaniumABILanguageRuntime_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/CPPLanguageRuntime.h"
#include "lldb/Core/Value.h"

namespace lldb_private {
    
    class ItaniumABILanguageRuntime :
    public lldb_private::CPPLanguageRuntime
    {
    public:
        ~ItaniumABILanguageRuntime() { }
        
        virtual bool
        IsVTableName (const char *name);
        
        //------------------------------------------------------------------
        // Static Functions
        //------------------------------------------------------------------
        static void
        Initialize();
        
        static void
        Terminate();
        
        static lldb_private::LanguageRuntime *
        CreateInstance (Process *process, lldb::LanguageType language);
        
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
        ItaniumABILanguageRuntime(Process *process) : lldb_private::CPPLanguageRuntime(process) { } // Call CreateInstance instead.
    };
    
} // namespace lldb_private

#endif  // liblldb_ItaniumABILanguageRuntime_h_
