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
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/Flags.h"

namespace lldb_private {

class CommandObject
{
public:

    typedef const char *(ArgumentHelpCallbackFunction) ();
    
    struct ArgumentHelpCallback
    {
        ArgumentHelpCallbackFunction  *help_callback;
        bool                           self_formatting;
        
        const char*
        operator () () const
        {
            return (*help_callback)();
        }
        
        operator bool() const
        {
            return (help_callback != NULL);
        }

    };
    
    struct ArgumentTableEntry  // Entries in the main argument information table
    {
        lldb::CommandArgumentType  arg_type;
        const char *arg_name;
        CommandCompletions::CommonCompletionTypes completion_type;
        ArgumentHelpCallback  help_function;
        const char *help_text;
    };
    
    struct CommandArgumentData  // Used to build individual command argument lists
    {
        lldb::CommandArgumentType arg_type;
        ArgumentRepetitionType arg_repetition;
    };
    
    typedef std::vector<CommandArgumentData> CommandArgumentEntry; // Used to build individual command argument lists

    static ArgumentTableEntry g_arguments_data[lldb::eArgTypeLastArg];   // Main argument information table

    typedef std::map<std::string, lldb::CommandObjectSP> CommandMap;

    CommandObject (CommandInterpreter &interpreter,
                   const char *name,
                   const char *help = NULL,
                   const char *syntax = NULL,
                   uint32_t flags = 0);

    virtual
    ~CommandObject ();

    
    static const char * 
    GetArgumentTypeAsCString (const lldb::CommandArgumentType arg_type);
    
    static const char * 
    GetArgumentDescriptionAsCString (const lldb::CommandArgumentType arg_type);

    CommandInterpreter &
    GetCommandInterpreter ()
    {
        return m_interpreter;
    }

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
    
    // override this to return true if you want to enable the user to delete
    // the Command object from the Command dictionary (aliases have their own
    // deletion scheme, so they do not need to care about this)
    virtual bool
    IsRemovable() { return false; }
    
    bool
    IsAlias () { return m_is_alias; }
    
    void
    SetIsAlias (bool value) { m_is_alias = value; }

    virtual bool
    IsMultiwordObject () { return false; }

    virtual bool
    WantsRawCommandString() { return false; }

    virtual Options *
    GetOptions ();

    static const ArgumentTableEntry*
    GetArgumentTable ();

    static lldb::CommandArgumentType
    LookupArgumentName (const char *arg_name);

    static ArgumentTableEntry *
    FindArgumentDataByType (lldb::CommandArgumentType arg_type);

    int
    GetNumArgumentEntries ();

    CommandArgumentEntry *
    GetArgumentEntryAtIndex (int idx);

    static void
    GetArgumentHelp (Stream &str, lldb::CommandArgumentType arg_type, CommandInterpreter &interpreter);

    static const char *
    GetArgumentName (lldb::CommandArgumentType arg_type);

    void
    GetFormattedCommandArguments (Stream &str);
    
    bool
    IsPairType (ArgumentRepetitionType arg_repeat_type);
    
    enum 
    {
        eFlagProcessMustBeLaunched = (1 << 0),
        eFlagProcessMustBePaused = (1 << 1)
    };

    // Do not override this
    bool
    ExecuteCommandString (const char *command,
                          CommandReturnObject &result);

    bool
    ParseOptions (Args& args,
                  CommandReturnObject &result);

    bool
    ExecuteWithOptions (Args& command,
                        CommandReturnObject &result);

    virtual bool
    ExecuteRawCommandString (const char *command,
                             CommandReturnObject &result)
    {
        return false;
    }


    virtual bool
    Execute (Args& command,
             CommandReturnObject &result) = 0;

    void
    SetCommandName (const char *name);

    // This function really deals with CommandObjectLists, but we didn't make a
    // CommandObjectList class, so I'm sticking it here.  But we really should have
    // such a class.  Anyway, it looks up the commands in the map that match the partial
    // string cmd_str, inserts the matches into matches, and returns the number added.

    static int
    AddNamesMatchingPartialString (CommandMap &in_map, const char *cmd_str, StringList &matches);

    //------------------------------------------------------------------
    /// The input array contains a parsed version of the line.  The insertion
    /// point is given by cursor_index (the index in input of the word containing
    /// the cursor) and cursor_char_position (the position of the cursor in that word.)
    /// This default version handles calling option argument completions and then calls
    /// HandleArgumentCompletion if the cursor is on an argument, not an option.
    /// Don't override this method, override HandleArgumentCompletion instead unless
    /// you have special reasons.
    ///
    /// @param[in] interpreter
    ///    The command interpreter doing the completion.
    ///
    /// @param[in] input
    ///    The command line parsed into words
    ///
    /// @param[in] cursor_index
    ///     The index in \ainput of the word in which the cursor lies.
    ///
    /// @param[in] cursor_char_pos
    ///     The character position of the cursor in its argument word.
    ///
    /// @param[in] match_start_point
    /// @param[in] match_return_elements
    ///     FIXME: Not yet implemented...  If there is a match that is expensive to compute, these are
    ///     here to allow you to compute the completions in batches.  Start the completion from \amatch_start_point,
    ///     and return \amatch_return_elements elements.
    ///
    /// @param[out] word_complete
    ///     \btrue if this is a complete option value (a space will be inserted after the
    ///     completion.)  \bfalse otherwise.
    ///
    /// @param[out] matches
    ///     The array of matches returned.
    ///
    /// FIXME: This is the wrong return value, since we also need to make a distinction between
    /// total number of matches, and the window the user wants returned.
    ///
    /// @return
    ///     \btrue if we were in an option, \bfalse otherwise.
    //------------------------------------------------------------------
    virtual int
    HandleCompletion (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      bool &word_complete,
                      StringList &matches);

    //------------------------------------------------------------------
    /// The input array contains a parsed version of the line.  The insertion
    /// point is given by cursor_index (the index in input of the word containing
    /// the cursor) and cursor_char_position (the position of the cursor in that word.)
    /// We've constructed the map of options and their arguments as well if that is
    /// helpful for the completion.
    ///
    /// @param[in] interpreter
    ///    The command interpreter doing the completion.
    ///
    /// @param[in] input
    ///    The command line parsed into words
    ///
    /// @param[in] cursor_index
    ///     The index in \ainput of the word in which the cursor lies.
    ///
    /// @param[in] cursor_char_pos
    ///     The character position of the cursor in its argument word.
    ///
    /// @param[in] opt_element_vector
    ///     The results of the options parse of \a input.
    ///
    /// @param[in] match_start_point
    /// @param[in] match_return_elements
    ///     See CommandObject::HandleCompletions for a description of how these work.
    ///
    /// @param[out] word_complete
    ///     \btrue if this is a complete option value (a space will be inserted after the
    ///     completion.)  \bfalse otherwise.
    ///
    /// @param[out] matches
    ///     The array of matches returned.
    ///
    /// FIXME: This is the wrong return value, since we also need to make a distinction between
    /// total number of matches, and the window the user wants returned.
    ///
    /// @return
    ///     \btrue if we were in an option, \bfalse otherwise.
    //------------------------------------------------------------------

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        return 0;
    }
    
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
    
    //------------------------------------------------------------------
    /// Get the command that appropriate for a "repeat" of the current command.
    ///
    /// @param[in] current_command_line
    ///    The complete current command line.
    ///
    /// @return
    ///     NULL if there is no special repeat command - it will use the current command line.
    ///     Otherwise a pointer to the command to be repeated.
    ///     If the returned string is the empty string, the command won't be repeated.    
    //------------------------------------------------------------------
    virtual const char *GetRepeatCommand (Args &current_command_args, uint32_t index)
    {
        return NULL;
    }

protected:
    CommandInterpreter &m_interpreter;
    std::string m_cmd_name;
    std::string m_cmd_help_short;
    std::string m_cmd_help_long;
    std::string m_cmd_syntax;
    bool m_is_alias;
    Flags       m_flags;
    std::vector<CommandArgumentEntry> m_arguments;
};

} // namespace lldb_private


#endif  // liblldb_CommandObject_h_
