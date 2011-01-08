//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdlib.h>

#include "lldb/API/SBBlock.h"
#include "lldb/API/SBCompileUnit.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBFunction.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBSymbol.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBProcess.h"

using namespace lldb;

//----------------------------------------------------------------------
// This quick sample code shows how to create a debugger instance and
// create an "i386" executable target. Then we can lookup the executable
// module and resolve a file address into a section offset address,
// and find all symbol context objects (if any) for that address: 
// compile unit, function, deepest block, line table entry and the 
// symbol.
//----------------------------------------------------------------------
int
main (int argc, char const *argv[])
{
    // Initialize LLDB
    SBDebugger::Initialize();

    if (argc < 3)
        exit (1);
    
    // The first argument is the file path we want to look something up in
    const char *exe_file_path = argv[1];
    // The second argument in the address that we want to lookup
    lldb::addr_t file_addr = strtoull (argv[2], NULL, 0);
    
    // Make a file spec out of our executable path
    SBFileSpec exe_file_spec (exe_file_path);
    
    // Create a debugger instance so we can create a target
    SBDebugger debugger (SBDebugger::Create());
    
    if (debugger.IsValid())
    {
        // Create a target using the executable.
        SBTarget target (debugger.CreateTargetWithFileAndArch (exe_file_path, "i386"));
        if (target.IsValid())
        {
            // Find the executable module so we can do a lookup inside it
            SBModule module (target.FindModule (exe_file_spec));
            SBAddress addr;
            
            // Take a file virtual address and resolve it to a section offset
            // address that can be used to do a symbol lookup by address
            if (module.ResolveFileAddress (file_addr, addr))
            {
                // We can resolve a section offset address in the module
                // and only ask for what we need. You can logical or together
                // bits from the SymbolContextItem enumeration found in 
                // lldb-enumeration.h to request only what you want. Here we
                // are asking for everything. 
                //
                // NOTE: the less you ask for, the less LLDB will parse as
                // LLDB does partial parsing on just about everything.
                SBSymbolContext symbol_context (module.ResolveSymbolContextForAddress (addr, eSymbolContextEverything));
                
                SBCompileUnit comp_unit (symbol_context.GetCompileUnit());
                if (comp_unit.IsValid())
                {
                }
                SBFunction function (symbol_context.GetFunction());
                if (function.IsValid())
                {
                }
                SBBlock block (symbol_context.GetBlock());
                if (block.IsValid())
                {
                }
                SBLineEntry line_entry (symbol_context.GetLineEntry());
                if (line_entry.IsValid())
                {
                }
                SBSymbol symbol (symbol_context.GetSymbol());
                if (symbol.IsValid())
                {
                }
            }
        }
    }

    // Terminate LLDB
    SBDebugger::Terminate();
    return 0;
}

