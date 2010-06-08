//===-- CommandObjectSource.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectSource.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Args.h"
#include "lldb/Interpreter/CommandContext.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/TargetList.h"

using namespace lldb;
using namespace lldb_private;

const char *k_space_characters = "\t\n\v\f\r ";

//-------------------------------------------------------------------------
// CommandObjectSource
//-------------------------------------------------------------------------

CommandObjectSource::CommandObjectSource() :
    CommandObject ("source",
                   "Reads in debugger commands from the file <filename> and executes them.",
                   "source <filename>")
{
}

CommandObjectSource::~CommandObjectSource ()
{
}

bool
CommandObjectSource::Execute
(
    Args& args,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    const int argc = args.GetArgumentCount();
    if (argc == 1)
    {
        const char *filename = args.GetArgumentAtIndex(0);
        bool success = true;

        result.AppendMessageWithFormat ("Executing commands in '%s'.\n", filename);

        FileSpec cmd_file (filename);
        if (cmd_file.Exists())
        {
            STLStringArray commands;
            success = cmd_file.ReadFileLines (commands);

            STLStringArray::iterator pos = commands.begin();

            // Trim out any empty lines or lines that start with the comment
            // char '#'
            while (pos != commands.end())
            {
                bool remove_string = false;
                size_t non_space = pos->find_first_not_of (k_space_characters);
                if (non_space == std::string::npos)
                    remove_string = true; // Empty line
                else if ((*pos)[non_space] == '#')
                    remove_string = true; // Comment line that starts with '#'

                if (remove_string)
                    pos = commands.erase(pos);
                else
                    ++pos;
            }

            if (commands.size() > 0)
            {
                const size_t num_commands = commands.size();
                size_t i;
                for (i = 0; i<num_commands; ++i)
                {
                    result.GetOutputStream().Printf("%s %s\n", interpreter->GetPrompt(), commands[i].c_str());
                    if (!interpreter->HandleCommand(commands[i].c_str(), false, result))
                        break;
                }

                if (i < num_commands)
                {
                    result.AppendErrorWithFormat("Aborting source of '%s' after command '%s' failed.\n", filename, commands[i].c_str());
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
                else
                {
                    success = true;
                    result.SetStatus (eReturnStatusFailed);
                }
            }
        }
        else
        {
            result.AppendErrorWithFormat ("File '%s' does not exist.\n", filename);
            result.SetStatus (eReturnStatusFailed);
            success = false;
        }

        if (success)
        {
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
    }
    else
    {
        result.AppendErrorWithFormat("'%s' takes exactly one executable filename argument.\n", GetCommandName());
        result.SetStatus (eReturnStatusFailed);
    }
    return result.Succeeded();

}
