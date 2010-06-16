//===-- CommandObjectBreakpoint.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectBreakpoint_h_
#define liblldb_CommandObjectBreakpoint_h_

// C Includes
// C++ Includes

#include <utility>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Address.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/STLUtils.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordBreakpoint
//-------------------------------------------------------------------------

class CommandObjectMultiwordBreakpoint : public CommandObjectMultiword
{
public:
    CommandObjectMultiwordBreakpoint (CommandInterpreter *interpreter);

    virtual
    ~CommandObjectMultiwordBreakpoint ();

    static void
    VerifyBreakpointIDs (Args &args, Target *target, CommandReturnObject &result, BreakpointIDList *valid_ids);

};

//-------------------------------------------------------------------------
// CommandObjectMultiwordBreakpointSet
//-------------------------------------------------------------------------


class CommandObjectBreakpointSet : public CommandObject
{
public:

    typedef enum BreakpointSetType
    {
        eSetTypeInvalid,
        eSetTypeFileAndLine,
        eSetTypeAddress,
        eSetTypeFunctionName,
        eSetTypeFunctionRegexp
    } BreakpointSetType;

    CommandObjectBreakpointSet ();

    virtual
    ~CommandObjectBreakpointSet ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
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

        std::string m_filename;
        unsigned int m_line_num;
        unsigned int m_column;
        bool m_ignore_inlines;
        std::string m_func_name;
        std::string m_func_regexp;
        lldb::addr_t m_load_addr;
        STLStringArray m_modules;
        int32_t m_ignore_count;
        lldb::tid_t m_thread_id;
        uint32_t m_thread_index;
        std::string m_thread_name;
        std::string m_queue_name;

    };

private:
    CommandOptions m_options;
};

//-------------------------------------------------------------------------
// CommandObjectMultiwordBreakpointConfigure
//-------------------------------------------------------------------------


class CommandObjectBreakpointConfigure : public CommandObject
{
public:

    CommandObjectBreakpointConfigure ();

    virtual
    ~CommandObjectBreakpointConfigure ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
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

        int32_t m_ignore_count;
        lldb::tid_t m_thread_id;
        uint32_t m_thread_index;
        std::string m_thread_name;
        std::string m_queue_name;

    };

private:
    CommandOptions m_options;
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointEnable
//-------------------------------------------------------------------------

class CommandObjectBreakpointEnable : public CommandObject
{
public:
    CommandObjectBreakpointEnable ();

    virtual
    ~CommandObjectBreakpointEnable ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);

private:
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointDisable
//-------------------------------------------------------------------------

class CommandObjectBreakpointDisable : public CommandObject
{
public:
    CommandObjectBreakpointDisable ();

    virtual
    ~CommandObjectBreakpointDisable ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);

private:
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointList
//-------------------------------------------------------------------------

class CommandObjectBreakpointList : public CommandObject
{
public:
    CommandObjectBreakpointList ();

    virtual
    ~CommandObjectBreakpointList ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
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

        const lldb::OptionDefinition *
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        lldb::DescriptionLevel m_level;

        bool m_internal;
    };

private:
    CommandOptions m_options;
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointDelete
//-------------------------------------------------------------------------

class CommandObjectBreakpointDelete : public CommandObject
{
public:
    CommandObjectBreakpointDelete ();

    virtual
    ~CommandObjectBreakpointDelete ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);

private:
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectBreakpoint_h_
