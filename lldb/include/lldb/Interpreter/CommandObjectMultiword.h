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
// These two want to iterate over the subcommand dictionary.
friend class CommandInterpreter;
friend class CommandObjectSyntax;
public:
    CommandObjectMultiword (CommandInterpreter &interpreter,
                            const char *name,
                            const char *help = NULL,
                            const char *syntax = NULL,
                            uint32_t flags = 0);
    
    virtual
    ~CommandObjectMultiword ();

    virtual bool
    IsMultiwordObject () { return true; }

    virtual bool
    LoadSubCommand (const char *cmd_name, 
                    const lldb::CommandObjectSP& command_obj);

    void
    GenerateHelpText (CommandReturnObject &result);

    lldb::CommandObjectSP
    GetSubcommandSP (const char *sub_cmd, StringList *matches = NULL);

    CommandObject *
    GetSubcommandObject (const char *sub_cmd, StringList *matches = NULL);

    virtual bool
    WantsRawCommandString() { return false; };

    virtual int
    HandleCompletion (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      bool &word_complete,
                      StringList &matches);

    virtual const char *GetRepeatCommand (Args &current_command_args, uint32_t index);

    virtual bool
    Execute (const char *args_string,
             CommandReturnObject &result);
    
    virtual bool
    IsRemovable() const { return m_can_be_removed; }
    
    void
    SetRemovable (bool removable)
    {
        m_can_be_removed = removable;
    }
    
protected:

    CommandObject::CommandMap m_subcommand_dict;
    bool m_can_be_removed;
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectMultiword_h_
