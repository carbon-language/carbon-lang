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
#include <sys/errno.h>
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

#pragma mark CommandObjectTargetImageSearchPaths

class CommandObjectTargetImageSearchPathsAdd : public CommandObject
{
public:

    CommandObjectTargetImageSearchPathsAdd () :
        CommandObject ("target image-search-paths add",
                       "Add new image search paths substitution pairs to the current target.",
                       "target image-search-paths add <path-prefix> <new-path-prefix> [<path-prefix> <new-path-prefix>] ...")
    {
    }

    ~CommandObjectTargetImageSearchPathsAdd ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            uint32_t argc = command.GetArgumentCount();
            if (argc & 1)
            {
                result.AppendError ("add requires an even number of arguments");
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
                    }
                    else
                    {
                        if (from[0])
                            result.AppendError ("<path-prefix> can't be empty");
                        else
                            result.AppendError ("<new-path-prefix> can't be empty");
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
            }
        }
        else
        {
            result.AppendError ("invalid target");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

class CommandObjectTargetImageSearchPathsClear : public CommandObject
{
public:

    CommandObjectTargetImageSearchPathsClear () :
        CommandObject ("target image-search-paths clear",
                       "Clears all current image search paths substitution pairs from the current target.",
                       "target image-search-paths clear")
    {
    }

    ~CommandObjectTargetImageSearchPathsClear ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            bool notify = true;
            target->GetImageSearchPathList().Clear(notify);
        }
        else
        {
            result.AppendError ("invalid target");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

class CommandObjectTargetImageSearchPathsInsert : public CommandObject
{
public:

    CommandObjectTargetImageSearchPathsInsert () :
        CommandObject ("target image-search-paths insert",
                       "Inserts a new image search paths substitution pair to the current target at the specified index.",
                       "target image-search-paths insert <index> <path-prefix> <new-path-prefix> [<path-prefix> <new-path-prefix>] ...")
    {
    }

    ~CommandObjectTargetImageSearchPathsInsert ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
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
                    }
                    else
                    {
                        if (from[0])
                            result.AppendError ("<path-prefix> can't be empty");
                        else
                            result.AppendError ("<new-path-prefix> can't be empty");
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                }
            }
            else
            {
                result.AppendError ("insert requires at least three arguments");
                result.SetStatus (eReturnStatusFailed);
                return result.Succeeded();
            }

        }
        else
        {
            result.AppendError ("invalid target");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

class CommandObjectTargetImageSearchPathsList : public CommandObject
{
public:

    CommandObjectTargetImageSearchPathsList () :
        CommandObject ("target image-search-paths list",
                       "Lists all current image search paths substitution pairs in the current target.",
                       "target image-search-paths list")
    {
    }

    ~CommandObjectTargetImageSearchPathsList ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            if (command.GetArgumentCount() != 0)
            {
                result.AppendError ("list takes no arguments");
                result.SetStatus (eReturnStatusFailed);
                return result.Succeeded();
            }

            target->GetImageSearchPathList().Dump(&result.GetOutputStream());
        }
        else
        {
            result.AppendError ("invalid target");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

class CommandObjectTargetImageSearchPathsQuery : public CommandObject
{
public:

    CommandObjectTargetImageSearchPathsQuery () :
    CommandObject ("target image-search-paths query",
                   "Transforms a path using the first applicable image search path.",
                   "target image-search-paths query <path>")
    {
    }

    ~CommandObjectTargetImageSearchPathsQuery ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
        if (target)
        {
            if (command.GetArgumentCount() != 1)
            {
                result.AppendError ("query requires one argument");
                result.SetStatus (eReturnStatusFailed);
                return result.Succeeded();
            }

            ConstString orig(command.GetArgumentAtIndex(0));
            ConstString transformed;
            if (target->GetImageSearchPathList().RemapPath(orig, transformed))
                result.GetOutputStream().Printf("%s\n", transformed.GetCString());
            else
                result.GetOutputStream().Printf("%s\n", orig.GetCString());
        }
        else
        {
            result.AppendError ("invalid target");
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
//    CommandObject ("frame select",
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
//             CommandInterpreter &interpreter,
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
//                                                             interpreter,
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


#pragma mark CommandObjectMultiwordTarget

//-------------------------------------------------------------------------
// CommandObjectMultiwordImageSearchPaths
//-------------------------------------------------------------------------

class CommandObjectMultiwordImageSearchPaths : public CommandObjectMultiword
{
public:

    CommandObjectMultiwordImageSearchPaths (CommandInterpreter &interpreter) :
        CommandObjectMultiword ("target image-search-paths",
                                "A set of commands for operating on debugger target image search paths.",
                                "target image-search-paths <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand (interpreter, "add",     CommandObjectSP (new CommandObjectTargetImageSearchPathsAdd ()));
        LoadSubCommand (interpreter, "clear",   CommandObjectSP (new CommandObjectTargetImageSearchPathsClear ()));
        LoadSubCommand (interpreter, "insert",  CommandObjectSP (new CommandObjectTargetImageSearchPathsInsert ()));
        LoadSubCommand (interpreter, "list",    CommandObjectSP (new CommandObjectTargetImageSearchPathsList ()));
        LoadSubCommand (interpreter, "query",   CommandObjectSP (new CommandObjectTargetImageSearchPathsQuery ()));
    }

    ~CommandObjectMultiwordImageSearchPaths()
    {
    }
};


#pragma mark CommandObjectMultiwordTarget

//-------------------------------------------------------------------------
// CommandObjectMultiwordTarget
//-------------------------------------------------------------------------

CommandObjectMultiwordTarget::CommandObjectMultiwordTarget (CommandInterpreter &interpreter) :
    CommandObjectMultiword ("target",
                            "A set of commands for operating on debugger targets.",
                            "target <subcommand> [<subcommand-options>]")
{
    LoadSubCommand (interpreter, "image-search-paths", CommandObjectSP (new CommandObjectMultiwordImageSearchPaths (interpreter)));
}

CommandObjectMultiwordTarget::~CommandObjectMultiwordTarget ()
{
}

