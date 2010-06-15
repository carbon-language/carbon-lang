//===-- CommandObject.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObject_h_
#define liblldb_CommandObject_h_

#include <map>
#include <set>
#include <string>
#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/Flags.h"

namespace lldb_private {

class CommandObject
{
public:
    typedef std::map<std::string, lldb::CommandObjectSP> CommandMap;


    CommandObject (const char *name,
                   const char *help = NULL,
                   const char *syntax = NULL,
                   uint32_t flags = 0);

    virtual
    ~CommandObject ();

    const char *
    GetHelp ();

    const char *
    GetHelpLong ();

    const char *
    GetSyntax ();

    const char *
    Translate ();

    const char *
    GetCommandName ();

    void
    SetHelp (const char * str);

    void
    SetHelpLong (const char * str);

    void
    SetSyntax (const char *str);

    virtual void
    AddObject (const char *obj_name) {}

    virtual bool
    IsCrossRefObject () { return false; }

    virtual bool
    IsMultiwordObject () { return false; }

    virtual bool
    WantsRawCommandString() { return false; }

    virtual Options *
    GetOptions ();

    enum 
    {
        eFlagProcessMustBeLaunched = (1 << 0),
        eFlagProcessMustBePaused = (1 << 1)
    };

    // Do not override this
    bool
    ExecuteCommandString (const char *command,
                          CommandContext *context,
                          CommandInterpreter *interpreter,
                          CommandReturnObject &result);

    bool
    ParseOptions(Args& args,
                 CommandInterpreter *interpreter,
                 CommandReturnObject &result);

    bool
    ExecuteWithOptions (Args& command,
                        CommandContext *context,
                        CommandInterpreter *interpreter,
                        CommandReturnObject &result);

    virtual bool
    ExecuteRawCommandString (const char *command,
                             CommandContext *context,
                             CommandInterpreter *interpreter,
                             CommandReturnObject &result)
    {
        return false;
    }


    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result) = 0;

    void
    SetCommandName (const char *name);

    // This function really deals with CommandObjectLists, but we didn't make a
    // CommandObjectList class, so I'm sticking it here.  But we really should have
    // such a class.  Anyway, it looks up the commands in the map that match the partial
    // string cmd_str, inserts the matches into matches, and returns the number added.

    static int
    AddNamesMatchingPartialString (CommandMap &in_map, const char *cmd_str, StringList &matches);

    // The input array contains a parsed version of the line.  The insertion
    // point is given by cursor_index (the index in input of the word containing
    // the cursor) and cursor_char_position (the position of the cursor in that word.)
    // This default version handles calling option argument completions and then calls
    // HandleArgumentCompletion if the cursor is on an argument, not an option.
    // Don't override this method, override HandleArgumentCompletion instead unless
    // you have special reasons.
    virtual int
    HandleCompletion (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      CommandInterpreter *interpreter,
                      StringList &matches);

    // The input array contains a parsed version of the line.  The insertion
    // point is given by cursor_index (the index in input of the word containing
    // the cursor) and cursor_char_position (the position of the cursor in that word.)
    // We've constructed the map of options and their arguments as well if that is
    // helpful for the completion.

    virtual int
    HandleArgumentCompletion (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      OptionElementVector &opt_element_vector,
                      int match_start_point,
                      int max_return_elements,
                      CommandInterpreter *interpreter,
                      StringList &matches);


    bool
    HelpTextContainsWord (const char *search_word);

    //------------------------------------------------------------------
    /// The flags accessor.
    ///
    /// @return
    ///     A reference to the Flags member variable.
    //------------------------------------------------------------------
    Flags&
    GetFlags();

    //------------------------------------------------------------------
    /// The flags const accessor.
    ///
    /// @return
    ///     A const reference to the Flags member variable.
    //------------------------------------------------------------------
    const Flags&
    GetFlags() const;

protected:
    std::string m_cmd_name;
    std::string m_cmd_help_short;
    std::string m_cmd_help_long;
    std::string m_cmd_syntax;
    Flags       m_flags;
};

} // namespace lldb_private


#endif  // liblldb_CommandObject_h_
