//===-- CommandObjectMultiword.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectMultiword_h_
#define liblldb_CommandObjectMultiword_h_

// C Includes
// C++ Includes
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiword
//-------------------------------------------------------------------------

class CommandObjectMultiword : public CommandObject
{
public:
    CommandObjectMultiword (const char *name,
                              const char *help = NULL,
                              const char *syntax = NULL,
                              uint32_t flags = 0);

    virtual
    ~CommandObjectMultiword ();

    virtual bool
    IsMultiwordObject () { return true; }

    bool
    LoadSubCommand (lldb::CommandObjectSP command_obj, const char *cmd_name, CommandInterpreter *interpreter);

    void
    GenerateHelpText (CommandReturnObject &result, CommandInterpreter *interpreter);

    lldb::CommandObjectSP
    GetSubcommandSP (const char *sub_cmd, StringList *matches = NULL);

    CommandObject *
    GetSubcommandObject (const char *sub_cmd, StringList *matches = NULL);

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);

    virtual int
    HandleCompletion (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      CommandInterpreter *interpreter,
                      StringList &matches);

    CommandObject::CommandMap m_subcommand_dict;
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectMultiword_h_
