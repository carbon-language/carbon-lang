//===-- CommandObjectSettings.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectSettings_h_
#define liblldb_CommandObjectSettings_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"


namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordSettings
//-------------------------------------------------------------------------

class CommandObjectMultiwordSettings : public CommandObjectMultiword
{
public:

    CommandObjectMultiwordSettings (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectMultiwordSettings ();

};

//-------------------------------------------------------------------------
// CommandObjectSettingsSet
//-------------------------------------------------------------------------

class CommandObjectSettingsSet : public CommandObject
{
public:
    CommandObjectSettingsSet ();

    virtual
    ~CommandObjectSettingsSet ();

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    virtual Options *
    GetOptions ();

    class CommandOptions : public Options
    {
    public:

        CommandOptions ();

        virtual
        ~CommandOptions ();

        virtual Error
        SetOptionValue (int option_idx, const char *option_arg);

        void
        ResetOptionValues ();

        const lldb::OptionDefinition*
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        bool m_override;
        bool m_reset;

    };

    virtual int
    HandleArgumentCompletion (CommandInterpreter &interpreter,
                              Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches);

private:
    CommandOptions m_options;
};

//-------------------------------------------------------------------------
// CommandObjectSettingsShow -- Show current values
//-------------------------------------------------------------------------

class CommandObjectSettingsShow : public CommandObject
{
public:
    CommandObjectSettingsShow ();

    virtual
    ~CommandObjectSettingsShow ();

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);


    virtual int
    HandleArgumentCompletion (CommandInterpreter &interpreter,
                              Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches);

private:
};

//-------------------------------------------------------------------------
// CommandObjectSettingsList -- List settable variables
//-------------------------------------------------------------------------

class CommandObjectSettingsList : public CommandObject
{
public: 
    CommandObjectSettingsList ();

    virtual
    ~CommandObjectSettingsList ();

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    virtual int
    HandleArgumentCompletion (CommandInterpreter &interpreter,
                              Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches);

private:
};

//-------------------------------------------------------------------------
// CommandObjectSettingsRemove
//-------------------------------------------------------------------------

class CommandObjectSettingsRemove : public CommandObject
{
public:
    CommandObjectSettingsRemove ();

    virtual
    ~CommandObjectSettingsRemove ();

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    virtual int
    HandleArgumentCompletion (CommandInterpreter &interpreter,
                              Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches);

private:
};

//-------------------------------------------------------------------------
// CommandObjectSettingsReplace
//-------------------------------------------------------------------------

class CommandObjectSettingsReplace : public CommandObject
{
public:
    CommandObjectSettingsReplace ();

    virtual
    ~CommandObjectSettingsReplace ();

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    virtual int
    HandleArgumentCompletion (CommandInterpreter &interpreter,
                              Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches);

private:
};

//-------------------------------------------------------------------------
// CommandObjectSettingsInsertBefore
//-------------------------------------------------------------------------

class CommandObjectSettingsInsertBefore : public CommandObject
{
public:
    CommandObjectSettingsInsertBefore ();

    virtual
    ~CommandObjectSettingsInsertBefore ();

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    virtual int
    HandleArgumentCompletion (CommandInterpreter &interpreter,
                              Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches);

private:
};

//-------------------------------------------------------------------------
// CommandObjectSettingInsertAfter
//-------------------------------------------------------------------------

class CommandObjectSettingsInsertAfter : public CommandObject
{
public:
    CommandObjectSettingsInsertAfter ();

    virtual
    ~CommandObjectSettingsInsertAfter ();

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    virtual int
    HandleArgumentCompletion (CommandInterpreter &interpreter,
                              Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches);

private:
};

//-------------------------------------------------------------------------
// CommandObjectSettingsAppend
//-------------------------------------------------------------------------

class CommandObjectSettingsAppend : public CommandObject
{
public:
    CommandObjectSettingsAppend ();

    virtual
    ~CommandObjectSettingsAppend ();

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    virtual int
    HandleArgumentCompletion (CommandInterpreter &interpreter,
                              Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches);

private:
};

//-------------------------------------------------------------------------
// CommandObjectSettingsClear
//-------------------------------------------------------------------------

class CommandObjectSettingsClear : public CommandObject
{
public:
    CommandObjectSettingsClear ();

    virtual
    ~CommandObjectSettingsClear ();

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    virtual int
    HandleArgumentCompletion (CommandInterpreter &interpreter,
                              Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches);

private:
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectSettings_h_
