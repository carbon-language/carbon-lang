//===-- CommandInterpreter.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandInterpreter_h_
#define liblldb_CommandInterpreter_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Core/Event.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/StringList.h"

namespace lldb_private {

class CommandInterpreter : public Broadcaster
{
public:
    typedef std::map<std::string, OptionArgVectorSP> OptionArgMap;

    enum
    {
        eBroadcastBitThreadShouldExit       = (1 << 0),
        eBroadcastBitResetPrompt            = (1 << 1),
        eBroadcastBitQuitCommandReceived    = (1 << 2)   // User entered quit
    };

    void
    SourceInitFile (bool in_cwd, CommandReturnObject &result);

    CommandInterpreter (Debugger &debugger,
                        lldb::ScriptLanguage script_language,
                        bool synchronous_execution);

    virtual
    ~CommandInterpreter ();

    lldb::CommandObjectSP
    GetCommandSPExact (const char *cmd, bool include_aliases);

    CommandObject *
    GetCommandObjectExact (const char *cmd_cstr, bool include_aliases);

    CommandObject *
    GetCommandObject (const char *cmd, StringList *matches = NULL);

    bool
    CommandExists (const char *cmd);

    bool
    AliasExists (const char *cmd);

    bool
    UserCommandExists (const char *cmd);

    void
    AddAlias (const char *alias_name, lldb::CommandObjectSP& command_obj_sp);

    bool
    RemoveAlias (const char *alias_name);

    bool
    RemoveUser (const char *alias_name);

    OptionArgVectorSP
    GetAliasOptions (const char *alias_name);

    void
    RemoveAliasOptions (const char *alias_name);

    void
    AddOrReplaceAliasOptions (const char *alias_name, OptionArgVectorSP &option_arg_vector_sp);

    bool
    HandleCommand (const char *command_line, 
                   bool add_to_history, 
                   CommandReturnObject &result, 
                   ExecutionContext *override_context = NULL);

    // This handles command line completion.  You are given a pointer to the command string buffer, to the current cursor,
    // and to the end of the string (in case it is not NULL terminated).
    // You also passed in an Args object to fill with the returns.
    // The first element of the array will be filled with the string that you would need to insert at
    // the cursor point to complete the cursor point to the longest common matching prefix.
    // If you want to limit the number of elements returned, set max_return_elements to the number of elements
    // you want returned.  Otherwise set max_return_elements to -1.
    // If you want to start some way into the match list, then set match_start_point to the desired start
    // point.
    // Returns the total number of completions, or -1 if the completion character should be inserted, or
    // INT_MAX if the number of matches is > max_return_elements, but it is expensive to compute.
    //
    // FIXME: Only max_return_elements == -1 is supported at present.

    int
    HandleCompletion (const char *current_line,
                                      const char *cursor,
                                      const char *last_char,
                                      int match_start_point,
                                      int max_return_elements,
                                      StringList &matches);

    // This version just returns matches, and doesn't compute the substring.  It is here so the
    // Help command can call it for the first argument.
    // word_complete tells whether a the completions are considered a "complete" response (so the
    // completer should complete the quote & put a space after the word.

    int
    HandleCompletionMatches (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      bool &word_complete,
                      StringList &matches);


    int
    GetCommandNamesMatchingPartialString (const char *cmd_cstr, bool include_aliases, StringList &matches);

    void
    GetHelp (CommandReturnObject &result);

    void
    GetAliasHelp (const char *alias_name, const char *command_name, StreamString &help_string);

    void
    OutputFormattedHelpText (Stream &stream,
                             const char *command_word,
                             const char *separator,
                             const char *help_text,
                             uint32_t max_word_len);

    Debugger &
    GetDebugger ()
    {
        return m_debugger;
    }

    const char *
    ProcessEmbeddedScriptCommands (const char *arg);

    const char *
    GetPrompt ();

    void
    SetPrompt (const char *);
    
    bool Confirm (const char *message, bool default_answer);
    
    static size_t
    GetConfirmationInputReaderCallback (void *baton,
                                        InputReader &reader,
                                        lldb::InputReaderAction action,
                                        const char *bytes,
                                        size_t bytes_len);
    
    void
    LoadCommandDictionary ();

    void
    Initialize ();

    void
    CrossRegisterCommand (const char * dest_cmd, const char * object_type);

    void
    SetScriptLanguage (lldb::ScriptLanguage lang);


    bool
    HasCommands ();

    bool
    HasAliases ();

    bool
    HasUserCommands ();

    bool
    HasAliasOptions ();

    void
    BuildAliasCommandArgs (CommandObject *alias_cmd_obj, const char *alias_name, Args &cmd_args, 
                           std::string &raw_input_string, CommandReturnObject &result);

    int
    GetOptionArgumentPosition (const char *in_string);

    ScriptInterpreter *
    GetScriptInterpreter ();

    void
    SkipLLDBInitFiles (bool skip_lldbinit_files)
    {
        m_skip_lldbinit_files = skip_lldbinit_files;
    }

    bool
    GetSynchronous ();

#ifndef SWIG
    void
    AddLogChannel (const char *name, const Log::Callbacks &log_callbacks);

    bool
    GetLogChannelCallbacks (const char *channel, Log::Callbacks &log_callbacks);

    bool
    RemoveLogChannel (const char *name);
#endif

    size_t
    FindLongestCommandWord (CommandObject::CommandMap &dict);

    void
    FindCommandsForApropos (const char *word, StringList &commands_found, StringList &commands_help);

    void
    AproposAllSubCommands (CommandObject *cmd_obj, const char *prefix, const char *search_word, 
                           StringList &commands_found, StringList &commands_help);

protected:
    friend class Debugger;

    void
    SetSynchronous (bool value);

    lldb::CommandObjectSP
    GetCommandSP (const char *cmd, bool include_aliases = true, bool exact = true, StringList *matches = NULL);

private:

    Debugger &m_debugger;   // The debugger session that this interpreter is associated with
    bool m_synchronous_execution;
    bool m_skip_lldbinit_files;
    CommandObject::CommandMap m_command_dict; // Stores basic built-in commands (they cannot be deleted, removed or overwritten).
    CommandObject::CommandMap m_alias_dict;   // Stores user aliases/abbreviations for commands
    CommandObject::CommandMap m_user_dict;    // Stores user-defined commands
    OptionArgMap m_alias_options; // Stores any options (with or without arguments) that go with any alias.
    std::vector<std::string> m_command_history;
    std::string m_repeat_command;  // Stores the command that will be executed for an empty command string.
};


} // namespace lldb_private

#endif  // liblldb_CommandInterpreter_h_
