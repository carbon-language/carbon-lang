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

#include <LLDB/LLDB.h>

using namespace lldb;

//----------------------------------------------------------------------
// This quick sample code shows how to create a debugger instance and
// create an executable target without adding dependent shared
// libraries. It will then set a regular expression breakpoint to get
// breakpoint locations for all functions in the module, and use the 
// locations to extract the symbol context for each location. Then it 
// dumps all // information about the function: its name, file address 
// range, the return type (if any), and all argument types.
//
// To build the program, type (while in this directory):
//
//    $ make
//
// then to run this on MacOSX, specify the path to your LLDB.framework
// library using the DYLD_FRAMEWORK_PATH option and run the executable
//
//    $ DYLD_FRAMEWORK_PATH=/Volumes/data/lldb/tot/build/Debug ./a.out executable_path1 [executable_path2 ...] 
//----------------------------------------------------------------------
class LLDBSentry
{
public:
    LLDBSentry() {
        // Initialize LLDB
        SBDebugger::Initialize();
    }
    ~LLDBSentry() {
        // Terminate LLDB
        SBDebugger::Terminate();
    }
};
int
main (int argc, char const *argv[])
{
    // Use a sentry object to properly initialize/terminate LLDB.
    LLDBSentry sentry;
    
    if (argc < 2)
        exit (1);
    
    const char *arch = NULL; // Fill this in with "x86_64" or "i386" as needed 
    const char *platform = NULL; // Leave NULL for native platform, set to a valid other platform name if required
    const bool add_dependent_libs = false;
    SBError error;
    for (int arg_idx = 1; arg_idx < argc; ++arg_idx)
    {
    // The first argument is the file path we want to look something up in
        const char *exe_file_path = argv[arg_idx];
        
        // Create a debugger instance so we can create a target
        SBDebugger debugger (SBDebugger::Create());
        
        if (debugger.IsValid())
        {
            // Create a target using the executable.
            SBTarget target = debugger.CreateTarget (exe_file_path,
                                                     arch,
                                                     platform,
                                                     add_dependent_libs,
                                                     error);
            
            if (error.Success())
            {
                if (target.IsValid())
                {
                    SBFileSpec exe_file_spec (exe_file_path, true);
                    SBModule module (target.FindModule (exe_file_spec));
                    SBFileSpecList comp_unit_list;

                    if (module.IsValid())
                    {
                        char command[1024];
                        lldb::SBCommandReturnObject command_result;
                        snprintf (command, sizeof(command), "add-dsym --uuid %s", module.GetUUIDString());
                        debugger.GetCommandInterpreter().HandleCommand (command, command_result);
                        if (!command_result.Succeeded())
                        {
                            fprintf (stderr, "error: couldn't locate debug symbols for '%s'\n", exe_file_path);
                            exit(1);
                        }

                        SBFileSpecList module_list;
                        module_list.Append(exe_file_spec);
                        SBBreakpoint bp = target.BreakpointCreateByRegex (".", module_list, comp_unit_list);
                        
                        const size_t num_locations = bp.GetNumLocations();
                        for (uint32_t bp_loc_idx=0; bp_loc_idx<num_locations; ++bp_loc_idx)
                        {
                            SBBreakpointLocation bp_loc = bp.GetLocationAtIndex(bp_loc_idx);
                            SBSymbolContext sc (bp_loc.GetAddress().GetSymbolContext(eSymbolContextEverything));
                            if (sc.IsValid())
                            {
                                if (sc.GetBlock().GetContainingInlinedBlock().IsValid())
                                {
                                    // Skip inlined functions
                                    continue;
                                }
                                SBFunction function (sc.GetFunction());
                                if (function.IsValid())
                                {
                                    addr_t lo_pc = function.GetStartAddress().GetFileAddress();
                                    if (lo_pc == LLDB_INVALID_ADDRESS)
                                    {
                                        // Skip functions that don't have concrete instances in the binary
                                        continue;
                                    }
                                    addr_t hi_pc = function.GetEndAddress().GetFileAddress();

                                    printf ("\nfunction name: %s\n", function.GetName());
                                    printf ("function range:[0x%llx - 0x%llx)\n", lo_pc, hi_pc);
                                    SBType function_type = function.GetType();
                                    SBType return_type = function_type.GetFunctionReturnType();
                                    if (return_type.IsValid())
                                    {
                                        printf ("return type: %s\n", return_type.GetName());
                                    }
                                    else
                                    {
                                        printf ("return type: <NONE>\n");
                                    }
                                    
                                    
                                    SBTypeList function_args = function_type.GetFunctionArgumentTypes();
                                    const size_t num_function_args = function_args.GetSize();
                                    for (uint32_t function_arg_idx = 0; function_arg_idx < num_function_args; ++function_arg_idx)
                                    {
                                        SBType function_arg_type = function_args.GetTypeAtIndex(function_arg_idx);
                                        if (function_arg_type.IsValid())
                                        {
                                            printf ("arg[%u] type: %s\n", function_arg_idx, function_arg_type.GetName());
                                        }
                                        else
                                        {
                                            printf ("arg[%u] type: <invalid>\n", function_arg_idx);
                                        }
                                    }
                                    
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                fprintf (stderr, "error: %s\n", error.GetCString());
                exit(1);
            }
        }
    }
    
    return 0;
}

