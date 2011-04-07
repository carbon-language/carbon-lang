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
    CommandObjectMultiwordBreakpoint (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectMultiwordBreakpoint ();

    static void
    VerifyBreakpointIDs (Args &args, Target *target, CommandReturnObject &result, BreakpointIDList *valid_ids);

};

//-------------------------------------------------------------------------
// CommandObjectdBreakpointSet
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

    CommandObjectBreakpointSet (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointSet ();

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
        SetOptionValue (int option_idx, const char *option_arg);

        void
        ResetOptionValues ();

        const OptionDefinition*
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        std::string m_filename;
        uint32_t m_line_num;
        uint32_t m_column;
        bool m_check_inlines;
        std::string m_func_name;
        uint32_t m_func_name_type_mask;
        std::string m_func_regexp;
        STLStringArray m_modules;
        lldb::addr_t m_load_addr;
        uint32_t m_ignore_count;
        lldb::tid_t m_thread_id;
        uint32_t m_thread_index;
        std::string m_thread_name;
        std::string m_queue_name;

    };

private:
    CommandOptions m_options;
};

//-------------------------------------------------------------------------
// CommandObjectMultiwordBreakpointModify
//-------------------------------------------------------------------------


class CommandObjectBreakpointModify : public CommandObject
{
public:

    CommandObjectBreakpointModify (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointModify ();

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
        SetOptionValue (int option_idx, const char *option_arg);

        void
        ResetOptionValues ();

        const OptionDefinition*
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        uint32_t m_ignore_count;
        lldb::tid_t m_thread_id;
        bool m_thread_id_passed;
        uint32_t m_thread_index;
        bool m_thread_index_passed;
        std::string m_thread_name;
        std::string m_queue_name;
        std::string m_condition;
        bool m_enable_passed;
        bool m_enable_value;
        bool m_name_passed;
        bool m_queue_passed;
        bool m_condition_passed;

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
    CommandObjectBreakpointEnable (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointEnable ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

private:
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointDisable
//-------------------------------------------------------------------------

class CommandObjectBreakpointDisable : public CommandObject
{
public:
    CommandObjectBreakpointDisable (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointDisable ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

private:
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointList
//-------------------------------------------------------------------------

class CommandObjectBreakpointList : public CommandObject
{
public:
    CommandObjectBreakpointList (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointList ();

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
        SetOptionValue (int option_idx, const char *option_arg);

        void
        ResetOptionValues ();

        const OptionDefinition *
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        lldb::DescriptionLevel m_level;

        bool m_internal;
    };

private:
    CommandOptions m_options;
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointClear
//-------------------------------------------------------------------------


class CommandObjectBreakpointClear : public CommandObject
{
public:

    typedef enum BreakpointClearType
    {
        eClearTypeInvalid,
        eClearTypeFileAndLine
    } BreakpointClearType;

    CommandObjectBreakpointClear (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointClear ();

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
        SetOptionValue (int option_idx, const char *option_arg);

        void
        ResetOptionValues ();

        const OptionDefinition*
        GetDefinitions ();

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        std::string m_filename;
        uint32_t m_line_num;

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
    CommandObjectBreakpointDelete (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectBreakpointDelete ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

private:
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectBreakpoint_h_
