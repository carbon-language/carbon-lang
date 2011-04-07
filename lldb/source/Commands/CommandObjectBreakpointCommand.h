//===-- CommandObjectBreakpointCommand.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectBreakpointCommand_h_
#define liblldb_CommandObjectBreakpointCommand_h_

// C Includes
// C++ Includes


// Other libraries and framework includes
// Project includes

#include "lldb/lldb-types.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"


namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordBreakpoint
//-------------------------------------------------------------------------

class CommandObjectBreakpointCommand : public CommandObjectMultiword
{
public:
    CommandObjectBreakpointCommand (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointCommand ();


    static bool
    BreakpointOptionsCallbackFunction (void *baton, 
                                       StoppointCallbackContext *context,
                                       lldb::user_id_t break_id, 
                                       lldb::user_id_t break_loc_id);
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointCommandAdd
//-------------------------------------------------------------------------


class CommandObjectBreakpointCommandAdd : public CommandObject
{
public:

    CommandObjectBreakpointCommandAdd (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointCommandAdd ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

    virtual Options *
    GetOptions ();

    void
    CollectDataForBreakpointCommandCallback (BreakpointOptions *bp_options, 
                                             CommandReturnObject &result);

    /// Set a one-liner as the callback for the breakpoint.
    void 
    SetBreakpointCommandCallback (BreakpointOptions *bp_options,
                                  const char *oneliner);

    static size_t
    GenerateBreakpointCommandCallback (void *baton, 
                                       InputReader &reader, 
                                       lldb::InputReaderAction notification,
                                       const char *bytes, 
                                       size_t bytes_len);
    
    static bool
    BreakpointOptionsCallbackFunction (void *baton, 
                                       StoppointCallbackContext *context, 
                                       lldb::user_id_t break_id,
                                       lldb::user_id_t break_loc_id);
    

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter);

        virtual
        ~CommandOptions ();

        virtual Error
        SetOptionValue (int option_idx, const char *option_arg);

        void
        ResetOptionValues ();

        const OptionDefinition*
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        bool m_use_commands;
        bool m_use_script_language;
        lldb::ScriptLanguage m_script_language;

        // Instance variables to hold the values for one_liner options.
        bool m_use_one_liner;
        std::string m_one_liner;
        bool m_stop_on_error;
    };

private:
    CommandOptions m_options;
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointCommandRemove
//-------------------------------------------------------------------------

class CommandObjectBreakpointCommandRemove : public CommandObject
{
public:
    CommandObjectBreakpointCommandRemove (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointCommandRemove ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

private:
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointCommandList
//-------------------------------------------------------------------------

class CommandObjectBreakpointCommandList : public CommandObject
{
public:
    CommandObjectBreakpointCommandList (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointCommandList ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

private:
};


} // namespace lldb_private

#endif  // liblldb_CommandObjectBreakpointCommand_h_
