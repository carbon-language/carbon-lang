//===-- CommandObjectTarget.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectTarget.h"

// C Includes
#include <errno.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/State.h"
#include "lldb/Core/Timer.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionGroupArchitecture.h"
#include "lldb/Interpreter/OptionGroupPlatform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;



static void
DumpTargetInfo (uint32_t target_idx, Target *target, const char *prefix_cstr, bool show_stopped_process_status, Stream &strm)
{
    const ArchSpec &target_arch = target->GetArchitecture();
    
    ModuleSP exe_module_sp (target->GetExecutableModule ());
    char exe_path[PATH_MAX];
    bool exe_valid = false;
    if (exe_module_sp)
        exe_valid = exe_module_sp->GetFileSpec().GetPath (exe_path, sizeof(exe_path));
    
    if (!exe_valid)
        ::strcpy (exe_path, "<none>");
    
    strm.Printf ("%starget #%u: %s", prefix_cstr ? prefix_cstr : "", target_idx, exe_path);
    
    uint32_t properties = 0;
    if (target_arch.IsValid())
    {
        strm.Printf ("%sarch=%s", properties++ > 0 ? ", " : " ( ", target_arch.GetTriple().str().c_str());
        properties++;
    }
    PlatformSP platform_sp (target->GetPlatform());
    if (platform_sp)
        strm.Printf ("%splatform=%s", properties++ > 0 ? ", " : " ( ", platform_sp->GetName());
    
    ProcessSP process_sp (target->GetProcessSP());
    bool show_process_status = false;
    if (process_sp)
    {
        lldb::pid_t pid = process_sp->GetID();
        StateType state = process_sp->GetState();
        if (show_stopped_process_status)
            show_process_status = StateIsStoppedState(state);
        const char *state_cstr = StateAsCString (state);
        if (pid != LLDB_INVALID_PROCESS_ID)
            strm.Printf ("%spid=%i", properties++ > 0 ? ", " : " ( ", pid);
        strm.Printf ("%sstate=%s", properties++ > 0 ? ", " : " ( ", state_cstr);
    }
    if (properties > 0)
        strm.PutCString (" )\n");
    else
        strm.EOL();
    if (show_process_status)
    {
        const bool only_threads_with_stop_reason = true;
        const uint32_t start_frame = 0;
        const uint32_t num_frames = 1;
        const uint32_t num_frames_with_source = 1;
        process_sp->GetStatus (strm);
        process_sp->GetThreadStatus (strm, 
                                     only_threads_with_stop_reason, 
                                     start_frame,
                                     num_frames,
                                     num_frames_with_source);

    }
}

static uint32_t
DumpTargetList (TargetList &target_list, bool show_stopped_process_status, Stream &strm)
{
    const uint32_t num_targets = target_list.GetNumTargets();
    if (num_targets)
    {        
        TargetSP selected_target_sp (target_list.GetSelectedTarget());
        strm.PutCString ("Current targets:\n");
        for (uint32_t i=0; i<num_targets; ++i)
        {
            TargetSP target_sp (target_list.GetTargetAtIndex (i));
            if (target_sp)
            {
                bool is_selected = target_sp.get() == selected_target_sp.get();
                DumpTargetInfo (i, 
                                target_sp.get(), 
                                is_selected ? "* " : "  ", 
                                show_stopped_process_status,
                                strm);
            }
        }
    }
    return num_targets;
}
#pragma mark CommandObjectTargetCreate

//-------------------------------------------------------------------------
// "target create"
//-------------------------------------------------------------------------

class CommandObjectTargetCreate : public CommandObject
{
public:
    CommandObjectTargetCreate(CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target create",
                       "Create a target using the argument as the main executable.",
                       NULL),
        m_option_group (interpreter),
        m_file_options (),
        m_platform_options(true) // Do include the "--platform" option in the platform settings by passing true
    {
        CommandArgumentEntry arg;
        CommandArgumentData file_arg;
    
        // Define the first (and only) variant of this arg.
            file_arg.arg_type = eArgTypeFilename;
        file_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (file_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
        
        m_option_group.Append (&m_file_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Append (&m_platform_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Finalize();
    }

    ~CommandObjectTargetCreate ()
    {
    }

    Options *
    GetOptions ()
    {
        return &m_option_group;
    }

    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const int argc = command.GetArgumentCount();
        if (argc == 1)
        {
            const char *file_path = command.GetArgumentAtIndex(0);
            Timer scoped_timer(__PRETTY_FUNCTION__, "(lldb) target create '%s'", file_path);
            FileSpec file_spec (file_path, true);
            
            bool select = true;
            PlatformSP platform_sp;
            
            Error error;
            
            if (m_platform_options.PlatformWasSpecified ())
            {
                platform_sp = m_platform_options.CreatePlatformWithOptions(m_interpreter, select, error);
                if (!platform_sp)
                {
                    result.AppendError(error.AsCString());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            ArchSpec file_arch;
            
            const char *arch_cstr = m_file_options.GetArchitectureName();
            if (arch_cstr)
            {        
                if (!platform_sp)
                    platform_sp = m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform();
                if (!m_file_options.GetArchitecture(platform_sp.get(), file_arch))
                {
                    result.AppendErrorWithFormat("invalid architecture '%s'\n", arch_cstr);
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            
            if (! file_spec.Exists() && !file_spec.ResolveExecutableLocation())
            {
                result.AppendErrorWithFormat ("File '%s' does not exist.\n", file_path);
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            
            TargetSP target_sp;
            Debugger &debugger = m_interpreter.GetDebugger();
            error = debugger.GetTargetList().CreateTarget (debugger, file_spec, file_arch, true, target_sp);
            
            if (target_sp)
            {
                debugger.GetTargetList().SetSelectedTarget(target_sp.get());
                result.AppendMessageWithFormat ("Current executable set to '%s' (%s).\n", file_path, target_sp->GetArchitecture().GetArchitectureName());
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
            else
            {
                result.AppendError(error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendErrorWithFormat("'%s' takes exactly one executable path argument.\n", m_cmd_name.c_str());
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
        
    }

    int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex(cursor_index));
        completion_str.erase (cursor_char_position);
        
        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter, 
                                                             CommandCompletions::eDiskFileCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }
private:
    OptionGroupOptions m_option_group;
    OptionGroupArchitecture m_file_options;
    OptionGroupPlatform m_platform_options;

};

#pragma mark CommandObjectTargetList

//----------------------------------------------------------------------
// "target list"
//----------------------------------------------------------------------

class CommandObjectTargetList : public CommandObject
{
public:
    CommandObjectTargetList (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target list",
                       "List all current targets in the current debug session.",
                       NULL,
                       0)
    {
    }
    
    virtual
    ~CommandObjectTargetList ()
    {
    }
    
    virtual bool
    Execute (Args& args, CommandReturnObject &result)
    {
        if (args.GetArgumentCount() == 0)
        {
            Stream &strm = result.GetOutputStream();
            
            bool show_stopped_process_status = false;
            if (DumpTargetList (m_interpreter.GetDebugger().GetTargetList(), show_stopped_process_status, strm) == 0)
            {
                strm.PutCString ("No targets.\n");
            }
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendError ("the 'target list' command takes no arguments\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};


#pragma mark CommandObjectTargetSelect

//----------------------------------------------------------------------
// "target select"
//----------------------------------------------------------------------

class CommandObjectTargetSelect : public CommandObject
{
public:
    CommandObjectTargetSelect (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target select",
                       "Select a target as the current target by target index.",
                       NULL,
                       0)
    {
    }
    
    virtual
    ~CommandObjectTargetSelect ()
    {
    }
    
    virtual bool
    Execute (Args& args, CommandReturnObject &result)
    {
        if (args.GetArgumentCount() == 1)
        {
            bool success = false;
            const char *target_idx_arg = args.GetArgumentAtIndex(0);
            uint32_t target_idx = Args::StringToUInt32 (target_idx_arg, UINT32_MAX, 0, &success);
            if (success)
            {
                TargetList &target_list = m_interpreter.GetDebugger().GetTargetList();
                const uint32_t num_targets = target_list.GetNumTargets();
                if (target_idx < num_targets)
                {        
                    TargetSP target_sp (target_list.GetTargetAtIndex (target_idx));
                    if (target_sp)
                    {
                        Stream &strm = result.GetOutputStream();
                        target_list.SetSelectedTarget (target_sp.get());
                        bool show_stopped_process_status = false;
                        DumpTargetList (target_list, show_stopped_process_status, strm);
                        result.SetStatus (eReturnStatusSuccessFinishResult);
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("target #%u is NULL in target list\n", target_idx);
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
                else
                {
                    result.AppendErrorWithFormat ("index %u is out of range, valid target indexes are 0 - %u\n", 
                                                  target_idx,
                                                  num_targets - 1);
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            else
            {
                result.AppendErrorWithFormat("invalid index string value '%s'\n", target_idx_arg);
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendError ("'target select' takes a single argument: a target index\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};


#pragma mark CommandObjectTargetImageSearchPaths

class CommandObjectTargetImageSearchPathsAdd : public CommandObject
{
public:

    CommandObjectTargetImageSearchPathsAdd (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target image-search-paths add",
                       "Add new image search paths substitution pairs to the current target.",
                       NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData old_prefix_arg;
        CommandArgumentData new_prefix_arg;
        
        // Define the first variant of this arg pair.
        old_prefix_arg.arg_type = eArgTypeOldPathPrefix;
        old_prefix_arg.arg_repetition = eArgRepeatPairPlus;
        
        // Define the first variant of this arg pair.
        new_prefix_arg.arg_type = eArgTypeNewPathPrefix;
        new_prefix_arg.arg_repetition = eArgRepeatPairPlus;
        
        // There are two required arguments that must always occur together, i.e. an argument "pair".  Because they
        // must always occur together, they are treated as two variants of one argument rather than two independent
        // arguments.  Push them both into the first argument position for m_arguments...

        arg.push_back (old_prefix_arg);
        arg.push_back (new_prefix_arg);

        m_arguments.push_back (arg);
    }

    ~CommandObjectTargetImageSearchPathsAdd ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            uint32_t argc = command.GetArgumentCount();
            if (argc & 1)
            {
                result.AppendError ("add requires an even number of arguments\n");
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                for (uint32_t i=0; i<argc; i+=2)
                {
                    const char *from = command.GetArgumentAtIndex(i);
                    const char *to = command.GetArgumentAtIndex(i+1);
                    
                    if (from[0] && to[0])
                    {
                        bool last_pair = ((argc - i) == 2);
                        target->GetImageSearchPathList().Append (ConstString(from),
                                                                 ConstString(to),
                                                                 last_pair); // Notify if this is the last pair
                        result.SetStatus (eReturnStatusSuccessFinishNoResult);
                    }
                    else
                    {
                        if (from[0])
                            result.AppendError ("<path-prefix> can't be empty\n");
                        else
                            result.AppendError ("<new-path-prefix> can't be empty\n");
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
            }
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

class CommandObjectTargetImageSearchPathsClear : public CommandObject
{
public:

    CommandObjectTargetImageSearchPathsClear (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target image-search-paths clear",
                       "Clear all current image search path substitution pairs from the current target.",
                       "target image-search-paths clear")
    {
    }

    ~CommandObjectTargetImageSearchPathsClear ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            bool notify = true;
            target->GetImageSearchPathList().Clear(notify);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

class CommandObjectTargetImageSearchPathsInsert : public CommandObject
{
public:

    CommandObjectTargetImageSearchPathsInsert (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target image-search-paths insert",
                       "Insert a new image search path substitution pair into the current target at the specified index.",
                       NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData index_arg;
        CommandArgumentData old_prefix_arg;
        CommandArgumentData new_prefix_arg;
        
        // Define the first and only variant of this arg.
        index_arg.arg_type = eArgTypeIndex;
        index_arg.arg_repetition = eArgRepeatPlain;

        // Put the one and only variant into the first arg for m_arguments:
        arg1.push_back (index_arg);

        // Define the first variant of this arg pair.
        old_prefix_arg.arg_type = eArgTypeOldPathPrefix;
        old_prefix_arg.arg_repetition = eArgRepeatPairPlus;
        
        // Define the first variant of this arg pair.
        new_prefix_arg.arg_type = eArgTypeNewPathPrefix;
        new_prefix_arg.arg_repetition = eArgRepeatPairPlus;
        
        // There are two required arguments that must always occur together, i.e. an argument "pair".  Because they
        // must always occur together, they are treated as two variants of one argument rather than two independent
        // arguments.  Push them both into the same argument position for m_arguments...

        arg2.push_back (old_prefix_arg);
        arg2.push_back (new_prefix_arg);

        // Add arguments to m_arguments.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    ~CommandObjectTargetImageSearchPathsInsert ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            uint32_t argc = command.GetArgumentCount();
            // check for at least 3 arguments and an odd nubmer of parameters
            if (argc >= 3 && argc & 1)
            {
                bool success = false;

                uint32_t insert_idx = Args::StringToUInt32(command.GetArgumentAtIndex(0), UINT32_MAX, 0, &success);

                if (!success)
                {
                    result.AppendErrorWithFormat("<index> parameter is not an integer: '%s'.\n", command.GetArgumentAtIndex(0));
                    result.SetStatus (eReturnStatusFailed);
                    return result.Succeeded();
                }

                // shift off the index
                command.Shift();
                argc = command.GetArgumentCount();

                for (uint32_t i=0; i<argc; i+=2, ++insert_idx)
                {
                    const char *from = command.GetArgumentAtIndex(i);
                    const char *to = command.GetArgumentAtIndex(i+1);
                    
                    if (from[0] && to[0])
                    {
                        bool last_pair = ((argc - i) == 2);
                        target->GetImageSearchPathList().Insert (ConstString(from),
                                                                 ConstString(to),
                                                                 insert_idx,
                                                                 last_pair);
                        result.SetStatus (eReturnStatusSuccessFinishNoResult);
                    }
                    else
                    {
                        if (from[0])
                            result.AppendError ("<path-prefix> can't be empty\n");
                        else
                            result.AppendError ("<new-path-prefix> can't be empty\n");
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                }
            }
            else
            {
                result.AppendError ("insert requires at least three arguments\n");
                result.SetStatus (eReturnStatusFailed);
                return result.Succeeded();
            }

        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

class CommandObjectTargetImageSearchPathsList : public CommandObject
{
public:

    CommandObjectTargetImageSearchPathsList (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target image-search-paths list",
                       "List all current image search path substitution pairs in the current target.",
                       "target image-search-paths list")
    {
    }

    ~CommandObjectTargetImageSearchPathsList ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            if (command.GetArgumentCount() != 0)
            {
                result.AppendError ("list takes no arguments\n");
                result.SetStatus (eReturnStatusFailed);
                return result.Succeeded();
            }

            target->GetImageSearchPathList().Dump(&result.GetOutputStream());
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

class CommandObjectTargetImageSearchPathsQuery : public CommandObject
{
public:

    CommandObjectTargetImageSearchPathsQuery (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "target image-search-paths query",
                   "Transform a path using the first applicable image search path.",
                   NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData path_arg;
        
        // Define the first (and only) variant of this arg.
        path_arg.arg_type = eArgTypePath;
        path_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (path_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    ~CommandObjectTargetImageSearchPathsQuery ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            if (command.GetArgumentCount() != 1)
            {
                result.AppendError ("query requires one argument\n");
                result.SetStatus (eReturnStatusFailed);
                return result.Succeeded();
            }

            ConstString orig(command.GetArgumentAtIndex(0));
            ConstString transformed;
            if (target->GetImageSearchPathList().RemapPath(orig, transformed))
                result.GetOutputStream().Printf("%s\n", transformed.GetCString());
            else
                result.GetOutputStream().Printf("%s\n", orig.GetCString());

            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

// TODO: implement the target select later when we start doing multiple targets
//#pragma mark CommandObjectTargetSelect
//
////-------------------------------------------------------------------------
//// CommandObjectTargetSelect
////-------------------------------------------------------------------------
//
//class CommandObjectTargetSelect : public CommandObject
//{
//public:
//
//    CommandObjectTargetSelect () :
//    CommandObject (interpreter,
//                   frame select",
//                   "Select the current frame by index in the current thread.",
//                   "frame select <frame-index>")
//    {
//    }
//
//    ~CommandObjectTargetSelect ()
//    {
//    }
//
//    bool
//    Execute (Args& command,
//             Debugger *context,
//             CommandInterpreter &m_interpreter,
//             CommandReturnObject &result)
//    {
//        ExecutionContext exe_ctx (context->GetExecutionContext());
//        if (exe_ctx.thread)
//        {
//            if (command.GetArgumentCount() == 1)
//            {
//                const char *frame_idx_cstr = command.GetArgumentAtIndex(0);
//
//                const uint32_t num_frames = exe_ctx.thread->GetStackFrameCount();
//                const uint32_t frame_idx = Args::StringToUInt32 (frame_idx_cstr, UINT32_MAX, 0);
//                if (frame_idx < num_frames)
//                {
//                    exe_ctx.thread->SetSelectedFrameByIndex (frame_idx);
//                    exe_ctx.frame = exe_ctx.thread->GetSelectedFrame ().get();
//
//                    if (exe_ctx.frame)
//                    {
//                        if (DisplayFrameForExecutionContext (exe_ctx.thread,
//                                                             exe_ctx.frame,
//                                                             m_interpreter,
//                                                             result.GetOutputStream(),
//                                                             true,
//                                                             true,
//                                                             3,
//                                                             3))
//                        {
//                            result.SetStatus (eReturnStatusSuccessFinishResult);
//                            return result.Succeeded();
//                        }
//                    }
//                }
//                if (frame_idx == UINT32_MAX)
//                    result.AppendErrorWithFormat ("Invalid frame index: %s.\n", frame_idx_cstr);
//                else
//                    result.AppendErrorWithFormat ("Frame index (%u) out of range.\n", frame_idx);
//            }
//            else
//            {
//                result.AppendError ("invalid arguments");
//                result.AppendErrorWithFormat ("Usage: %s\n", m_cmd_syntax.c_str());
//            }
//        }
//        else
//        {
//            result.AppendError ("no current thread");
//        }
//        result.SetStatus (eReturnStatusFailed);
//        return false;
//    }
//};


#pragma mark CommandObjectMultiwordImageSearchPaths

//-------------------------------------------------------------------------
// CommandObjectMultiwordImageSearchPaths
//-------------------------------------------------------------------------

class CommandObjectMultiwordImageSearchPaths : public CommandObjectMultiword
{
public:

    CommandObjectMultiwordImageSearchPaths (CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter, 
                                "target image-search-paths",
                                "A set of commands for operating on debugger target image search paths.",
                                "target image-search-paths <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand ("add",     CommandObjectSP (new CommandObjectTargetImageSearchPathsAdd (interpreter)));
        LoadSubCommand ("clear",   CommandObjectSP (new CommandObjectTargetImageSearchPathsClear (interpreter)));
        LoadSubCommand ("insert",  CommandObjectSP (new CommandObjectTargetImageSearchPathsInsert (interpreter)));
        LoadSubCommand ("list",    CommandObjectSP (new CommandObjectTargetImageSearchPathsList (interpreter)));
        LoadSubCommand ("query",   CommandObjectSP (new CommandObjectTargetImageSearchPathsQuery (interpreter)));
    }

    ~CommandObjectMultiwordImageSearchPaths()
    {
    }
};

#pragma mark CommandObjectTargetStopHookAdd

//-------------------------------------------------------------------------
// CommandObjectTargetStopHookAdd
//-------------------------------------------------------------------------

class CommandObjectTargetStopHookAdd : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:
        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter),
            m_line_start(0),
            m_line_end (UINT_MAX),
            m_func_name_type_mask (eFunctionNameTypeAuto),
            m_sym_ctx_specified (false),
            m_thread_specified (false),
            m_use_one_liner (false),
            m_one_liner()
        {
        }
        
        ~CommandOptions () {}
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            bool success;

            switch (short_option)
            {
                case 'c':
                    m_class_name = option_arg;
                    m_sym_ctx_specified = true;
                break;
                
                case 'e':
                    m_line_end = Args::StringToUInt32 (option_arg, UINT_MAX, 0, &success);
                    if (!success)
                    {
                        error.SetErrorStringWithFormat ("Invalid end line number: \"%s\".", option_arg);
                        break;
                    }
                    m_sym_ctx_specified = true;
                break;
                
                case 'l':
                    m_line_start = Args::StringToUInt32 (option_arg, 0, 0, &success);
                    if (!success)
                    {
                        error.SetErrorStringWithFormat ("Invalid start line number: \"%s\".", option_arg);
                        break;
                    }
                    m_sym_ctx_specified = true;
                break;
                
                case 'n':
                    m_function_name = option_arg;
                    m_func_name_type_mask |= eFunctionNameTypeAuto;
                    m_sym_ctx_specified = true;
                break;
                
                case 'f':
                    m_file_name = option_arg;
                    m_sym_ctx_specified = true;
                break;
                case 's':
                    m_module_name = option_arg;
                    m_sym_ctx_specified = true;
                break;
                case 't' :
                {
                    m_thread_id = Args::StringToUInt64(option_arg, LLDB_INVALID_THREAD_ID, 0);
                    if (m_thread_id == LLDB_INVALID_THREAD_ID)
                       error.SetErrorStringWithFormat ("Invalid thread id string '%s'.\n", option_arg);
                    m_thread_specified = true;
                }
                break;
                case 'T':
                    m_thread_name = option_arg;
                    m_thread_specified = true;
                break;
                case 'q':
                    m_queue_name = option_arg;
                    m_thread_specified = true;
                    break;
                case 'x':
                {
                    m_thread_index = Args::StringToUInt32(option_arg, UINT32_MAX, 0);
                    if (m_thread_id == UINT32_MAX)
                       error.SetErrorStringWithFormat ("Invalid thread index string '%s'.\n", option_arg);
                    m_thread_specified = true;
                }
                break;
                case 'o':
                    m_use_one_liner = true;
                    m_one_liner = option_arg;
                break;
                default:
                    error.SetErrorStringWithFormat ("Unrecognized option %c.");
                break;
            }
            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_class_name.clear();
            m_function_name.clear();
            m_line_start = 0;
            m_line_end = UINT_MAX;
            m_file_name.clear();
            m_module_name.clear();
            m_func_name_type_mask = eFunctionNameTypeAuto;
            m_thread_id = LLDB_INVALID_THREAD_ID;
            m_thread_index = UINT32_MAX;
            m_thread_name.clear();
            m_queue_name.clear();

            m_sym_ctx_specified = false;
            m_thread_specified = false;

            m_use_one_liner = false;
            m_one_liner.clear();
        }

        
        static OptionDefinition g_option_table[];
        
        std::string m_class_name;
        std::string m_function_name;
        uint32_t    m_line_start;
        uint32_t    m_line_end;
        std::string m_file_name;
        std::string m_module_name;
        uint32_t m_func_name_type_mask;  // A pick from lldb::FunctionNameType.
        lldb::tid_t m_thread_id;
        uint32_t m_thread_index;
        std::string m_thread_name;
        std::string m_queue_name;
        bool        m_sym_ctx_specified;
        bool        m_thread_specified;
        // Instance variables to hold the values for one_liner options.
        bool m_use_one_liner;
        std::string m_one_liner;
    };

    Options *
    GetOptions ()
    {
        return &m_options;
    }

    CommandObjectTargetStopHookAdd (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target stop-hook add ",
                       "Add a hook to be executed when the target stops.",
                       "target stop-hook add"),
        m_options (interpreter)
    {
    }

    ~CommandObjectTargetStopHookAdd ()
    {
    }

    static size_t 
    ReadCommandsCallbackFunction (void *baton, 
                                  InputReader &reader, 
                                  lldb::InputReaderAction notification,
                                  const char *bytes, 
                                  size_t bytes_len)
    {
        File &out_file = reader.GetDebugger().GetOutputFile();
        Target::StopHook *new_stop_hook = ((Target::StopHook *) baton);

        switch (notification)
        {
        case eInputReaderActivate:
            out_file.Printf ("%s\n", "Enter your stop hook command(s).  Type 'DONE' to end.");
            if (reader.GetPrompt())
                out_file.Printf ("%s", reader.GetPrompt());
            out_file.Flush();
            break;

        case eInputReaderDeactivate:
            break;

        case eInputReaderReactivate:
            if (reader.GetPrompt())
            {
                out_file.Printf ("%s", reader.GetPrompt());
                out_file.Flush();
            }
            break;

        case eInputReaderAsynchronousOutputWritten:
            break;
            
        case eInputReaderGotToken:
            if (bytes && bytes_len && baton)
            {
                StringList *commands = new_stop_hook->GetCommandPointer();
                if (commands)
                {
                    commands->AppendString (bytes, bytes_len); 
                }
            }
            if (!reader.IsDone() && reader.GetPrompt())
            {
                out_file.Printf ("%s", reader.GetPrompt());
                out_file.Flush();
            }
            break;
            
        case eInputReaderInterrupt:
            {
                // Finish, and cancel the stop hook.
                new_stop_hook->GetTarget()->RemoveStopHookByID(new_stop_hook->GetID());
                out_file.Printf ("Stop hook cancelled.\n");

                reader.SetIsDone (true);
            }
            break;
            
        case eInputReaderEndOfFile:
            reader.SetIsDone (true);
            break;
            
        case eInputReaderDone:
            out_file.Printf ("Stop hook #%d added.\n", new_stop_hook->GetID());
            break;
        }

        return bytes_len;
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            Target::StopHookSP new_hook_sp;
            target->AddStopHook (new_hook_sp);

            //  First step, make the specifier.
            std::auto_ptr<SymbolContextSpecifier> specifier_ap;
            if (m_options.m_sym_ctx_specified)
            {
                specifier_ap.reset(new SymbolContextSpecifier(m_interpreter.GetDebugger().GetSelectedTarget()));
                
                if (!m_options.m_module_name.empty())
                {
                    specifier_ap->AddSpecification (m_options.m_module_name.c_str(), SymbolContextSpecifier::eModuleSpecified);
                }
                
                if (!m_options.m_class_name.empty())
                {
                    specifier_ap->AddSpecification (m_options.m_class_name.c_str(), SymbolContextSpecifier::eClassOrNamespaceSpecified);
                }
                
                if (!m_options.m_file_name.empty())
                {
                    specifier_ap->AddSpecification (m_options.m_file_name.c_str(), SymbolContextSpecifier::eFileSpecified);
                }
                
                if (m_options.m_line_start != 0)
                {
                    specifier_ap->AddLineSpecification (m_options.m_line_start, SymbolContextSpecifier::eLineStartSpecified);
                }
                
                if (m_options.m_line_end != UINT_MAX)
                {
                    specifier_ap->AddLineSpecification (m_options.m_line_end, SymbolContextSpecifier::eLineEndSpecified);
                }
                
                if (!m_options.m_function_name.empty())
                {
                    specifier_ap->AddSpecification (m_options.m_function_name.c_str(), SymbolContextSpecifier::eFunctionSpecified);
                }
            }
            
            if (specifier_ap.get())
                new_hook_sp->SetSpecifier (specifier_ap.release());

            // Next see if any of the thread options have been entered:
            
            if (m_options.m_thread_specified)
            {
                ThreadSpec *thread_spec = new ThreadSpec();
                
                if (m_options.m_thread_id != LLDB_INVALID_THREAD_ID)
                {
                    thread_spec->SetTID (m_options.m_thread_id);
                }
                
                if (m_options.m_thread_index != UINT32_MAX)
                    thread_spec->SetIndex (m_options.m_thread_index);
                
                if (!m_options.m_thread_name.empty())
                    thread_spec->SetName (m_options.m_thread_name.c_str());
                
                if (!m_options.m_queue_name.empty())
                    thread_spec->SetQueueName (m_options.m_queue_name.c_str());
                    
                new_hook_sp->SetThreadSpecifier (thread_spec);
            
            }
            if (m_options.m_use_one_liner)
            {
                // Use one-liner.
                new_hook_sp->GetCommandPointer()->AppendString (m_options.m_one_liner.c_str());
                result.AppendMessageWithFormat("Stop hook #%d added.\n", new_hook_sp->GetID());
            }
            else
            {
                // Otherwise gather up the command list, we'll push an input reader and suck the data from that directly into
                // the new stop hook's command string.
                InputReaderSP reader_sp (new InputReader(m_interpreter.GetDebugger()));
                if (!reader_sp)
                {
                    result.AppendError("out of memory\n");
                    result.SetStatus (eReturnStatusFailed);
                    target->RemoveStopHookByID (new_hook_sp->GetID());
                    return false;
                }
                
                Error err (reader_sp->Initialize (CommandObjectTargetStopHookAdd::ReadCommandsCallbackFunction,
                                                  new_hook_sp.get(), // baton
                                                  eInputReaderGranularityLine,  // token size, to pass to callback function
                                                  "DONE",                       // end token
                                                  "> ",                         // prompt
                                                  true));                       // echo input
                if (!err.Success())
                {
                    result.AppendError (err.AsCString());
                    result.SetStatus (eReturnStatusFailed);
                    target->RemoveStopHookByID (new_hook_sp->GetID());
                    return false;
                }
                m_interpreter.GetDebugger().PushInputReader (reader_sp);
            }
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        
        return result.Succeeded();
    }
private:
    CommandOptions m_options;
};

OptionDefinition
CommandObjectTargetStopHookAdd::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "one-liner", 'o', required_argument, NULL, NULL, eArgTypeOneLiner,
        "Specify a one-line breakpoint command inline. Be sure to surround it with quotes." },
    { LLDB_OPT_SET_ALL, false, "shlib", 's', required_argument, NULL, CommandCompletions::eModuleCompletion, eArgTypeShlibName,
        "Set the module within which the stop-hook is to be run."},
    { LLDB_OPT_SET_ALL, false, "thread-index", 'x', required_argument, NULL, NULL, eArgTypeThreadIndex,
        "The stop hook is run only for the thread whose index matches this argument."},
    { LLDB_OPT_SET_ALL, false, "thread-id", 't', required_argument, NULL, NULL, eArgTypeThreadID,
        "The stop hook is run only for the thread whose TID matches this argument."},
    { LLDB_OPT_SET_ALL, false, "thread-name", 'T', required_argument, NULL, NULL, eArgTypeThreadName,
        "The stop hook is run only for the thread whose thread name matches this argument."},
    { LLDB_OPT_SET_ALL, false, "queue-name", 'q', required_argument, NULL, NULL, eArgTypeQueueName,
        "The stop hook is run only for threads in the queue whose name is given by this argument."},
    { LLDB_OPT_SET_1, false, "file", 'f', required_argument, NULL, CommandCompletions::eSourceFileCompletion, eArgTypeFilename,
        "Specify the source file within which the stop-hook is to be run." },
    { LLDB_OPT_SET_1, false, "start-line", 'l', required_argument, NULL, 0, eArgTypeLineNum,
        "Set the start of the line range for which the stop-hook is to be run."},
    { LLDB_OPT_SET_1, false, "end-line", 'e', required_argument, NULL, 0, eArgTypeLineNum,
        "Set the end of the line range for which the stop-hook is to be run."},
    { LLDB_OPT_SET_2, false, "classname", 'c', required_argument, NULL, NULL, eArgTypeClassName,
        "Specify the class within which the stop-hook is to be run." },
    { LLDB_OPT_SET_3, false, "name", 'n', required_argument, NULL, CommandCompletions::eSymbolCompletion, eArgTypeFunctionName,
        "Set the function name within which the stop hook will be run." },
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

#pragma mark CommandObjectTargetStopHookDelete

//-------------------------------------------------------------------------
// CommandObjectTargetStopHookDelete
//-------------------------------------------------------------------------

class CommandObjectTargetStopHookDelete : public CommandObject
{
public:

    CommandObjectTargetStopHookDelete (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target stop-hook delete [<id>]",
                       "Delete a stop-hook.",
                       "target stop-hook delete")
    {
    }

    ~CommandObjectTargetStopHookDelete ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            // FIXME: see if we can use the breakpoint id style parser?
            size_t num_args = command.GetArgumentCount();
            if (num_args == 0)
            {
                if (!m_interpreter.Confirm ("Delete all stop hooks?", true))
                {
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                else
                {
                    target->RemoveAllStopHooks();
                }
            }
            else
            {
                bool success;
                for (size_t i = 0; i < num_args; i++)
                {
                    lldb::user_id_t user_id = Args::StringToUInt32 (command.GetArgumentAtIndex(i), 0, 0, &success);
                    if (!success)
                    {
                        result.AppendErrorWithFormat ("invalid stop hook id: \"%s\".\n", command.GetArgumentAtIndex(i));
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                    success = target->RemoveStopHookByID (user_id);
                    if (!success)
                    {
                        result.AppendErrorWithFormat ("unknown stop hook id: \"%s\".\n", command.GetArgumentAtIndex(i));
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                }
            }
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        
        return result.Succeeded();
    }
};
#pragma mark CommandObjectTargetStopHookEnableDisable

//-------------------------------------------------------------------------
// CommandObjectTargetStopHookEnableDisable
//-------------------------------------------------------------------------

class CommandObjectTargetStopHookEnableDisable : public CommandObject
{
public:

    CommandObjectTargetStopHookEnableDisable (CommandInterpreter &interpreter, bool enable, const char *name, const char *help, const char *syntax) :
        CommandObject (interpreter,
                       name,
                       help,
                       syntax),
        m_enable (enable)
    {
    }

    ~CommandObjectTargetStopHookEnableDisable ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            // FIXME: see if we can use the breakpoint id style parser?
            size_t num_args = command.GetArgumentCount();
            bool success;
            
            if (num_args == 0)
            {
                target->SetAllStopHooksActiveState (m_enable);
            }
            else
            {
                for (size_t i = 0; i < num_args; i++)
                {
                    lldb::user_id_t user_id = Args::StringToUInt32 (command.GetArgumentAtIndex(i), 0, 0, &success);
                    if (!success)
                    {
                        result.AppendErrorWithFormat ("invalid stop hook id: \"%s\".\n", command.GetArgumentAtIndex(i));
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                    success = target->SetStopHookActiveStateByID (user_id, m_enable);
                    if (!success)
                    {
                        result.AppendErrorWithFormat ("unknown stop hook id: \"%s\".\n", command.GetArgumentAtIndex(i));
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                }
            }
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
private:
    bool m_enable;
};

#pragma mark CommandObjectTargetStopHookList

//-------------------------------------------------------------------------
// CommandObjectTargetStopHookList
//-------------------------------------------------------------------------

class CommandObjectTargetStopHookList : public CommandObject
{
public:

    CommandObjectTargetStopHookList (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "target stop-hook list [<type>]",
                       "List all stop-hooks.",
                       "target stop-hook list")
    {
    }

    ~CommandObjectTargetStopHookList ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            bool notify = true;
            target->GetImageSearchPathList().Clear(notify);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("invalid target\n");
            result.SetStatus (eReturnStatusFailed);
        }
        
        size_t num_hooks = target->GetNumStopHooks ();
        if (num_hooks == 0)
        {
            result.GetOutputStream().PutCString ("No stop hooks.\n");
        }
        else
        {
            for (size_t i = 0; i < num_hooks; i++)
            {
                Target::StopHookSP this_hook = target->GetStopHookAtIndex (i);
                if (i > 0)
                    result.GetOutputStream().PutCString ("\n");
                this_hook->GetDescription (&(result.GetOutputStream()), eDescriptionLevelFull);
            }
        }
        return result.Succeeded();
    }
};

#pragma mark CommandObjectMultiwordTargetStopHooks
//-------------------------------------------------------------------------
// CommandObjectMultiwordTargetStopHooks
//-------------------------------------------------------------------------

class CommandObjectMultiwordTargetStopHooks : public CommandObjectMultiword
{
public:

    CommandObjectMultiwordTargetStopHooks (CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter, 
                                "target stop-hook",
                                "A set of commands for operating on debugger target stop-hooks.",
                                "target stop-hook <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand ("add",      CommandObjectSP (new CommandObjectTargetStopHookAdd (interpreter)));
        LoadSubCommand ("delete",   CommandObjectSP (new CommandObjectTargetStopHookDelete (interpreter)));
        LoadSubCommand ("disable",  CommandObjectSP (new CommandObjectTargetStopHookEnableDisable (interpreter, 
                                                                                                   false, 
                                                                                                   "target stop-hook disable [<id>]",
                                                                                                   "Disable a stop-hook.",
                                                                                                   "target stop-hook disable")));
        LoadSubCommand ("enable",   CommandObjectSP (new CommandObjectTargetStopHookEnableDisable (interpreter, 
                                                                                                   true, 
                                                                                                   "target stop-hook enable [<id>]",
                                                                                                   "Enable a stop-hook.",
                                                                                                   "target stop-hook enable")));
        LoadSubCommand ("list",     CommandObjectSP (new CommandObjectTargetStopHookList (interpreter)));
    }

    ~CommandObjectMultiwordTargetStopHooks()
    {
    }
};



#pragma mark CommandObjectMultiwordTarget

//-------------------------------------------------------------------------
// CommandObjectMultiwordTarget
//-------------------------------------------------------------------------

CommandObjectMultiwordTarget::CommandObjectMultiwordTarget (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "target",
                            "A set of commands for operating on debugger targets.",
                            "target <subcommand> [<subcommand-options>]")
{
    
    LoadSubCommand ("create",    CommandObjectSP (new CommandObjectTargetCreate (interpreter)));
    LoadSubCommand ("list",      CommandObjectSP (new CommandObjectTargetList   (interpreter)));
    LoadSubCommand ("select",    CommandObjectSP (new CommandObjectTargetSelect (interpreter)));
    LoadSubCommand ("stop-hook", CommandObjectSP (new CommandObjectMultiwordTargetStopHooks (interpreter)));
    LoadSubCommand ("image-search-paths", CommandObjectSP (new CommandObjectMultiwordImageSearchPaths (interpreter)));
}

CommandObjectMultiwordTarget::~CommandObjectMultiwordTarget ()
{
}


