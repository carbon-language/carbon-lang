//===-- CommandObjectWatchpoint.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectWatchpoint_h_
#define liblldb_CommandObjectWatchpoint_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordWatchpoint
//-------------------------------------------------------------------------

class CommandObjectMultiwordWatchpoint : public CommandObjectMultiword
{
public:
    CommandObjectMultiwordWatchpoint (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectMultiwordWatchpoint ();
};

//-------------------------------------------------------------------------
// CommandObjectWatchpointList
//-------------------------------------------------------------------------

class CommandObjectWatchpointList : public CommandObject
{
public:
    CommandObjectWatchpointList (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectWatchpointList ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

    virtual Options *
    GetOptions ();

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter);

        virtual
        ~CommandOptions ();

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg);

        void
        OptionParsingStarting ();

        const OptionDefinition *
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        lldb::DescriptionLevel m_level;
    };

private:
    CommandOptions m_options;
};

//-------------------------------------------------------------------------
// CommandObjectWatchpointEnable
//-------------------------------------------------------------------------

class CommandObjectWatchpointEnable : public CommandObject
{
public:
    CommandObjectWatchpointEnable (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectWatchpointEnable ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

private:
};

//-------------------------------------------------------------------------
// CommandObjectWatchpointDisable
//-------------------------------------------------------------------------

class CommandObjectWatchpointDisable : public CommandObject
{
public:
    CommandObjectWatchpointDisable (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectWatchpointDisable ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

private:
};

//-------------------------------------------------------------------------
// CommandObjectWatchpointDelete
//-------------------------------------------------------------------------

class CommandObjectWatchpointDelete : public CommandObject
{
public:
    CommandObjectWatchpointDelete (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectWatchpointDelete ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

private:
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectWatchpoint_h_
