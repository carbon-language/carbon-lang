//===-- CommandObjectType.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectType.h"

// C Includes
// C++ Includes

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/FormatManager.h"
#include "lldb/Core/State.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectTypeFormatAdd
//-------------------------------------------------------------------------

class CommandObjectTypeFormatAdd : public CommandObject
{
    
private:
    
    class CommandOptions : public Options
    {
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
        Options (interpreter)
        {
        }
        
        virtual
        ~CommandOptions (){}
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            bool success;
            
            switch (short_option)
            {
                case 'c':
                    m_cascade = Args::StringToBoolean(option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for cascade: %s.\n", option_arg);
                    break;
                default:
                    error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
                    break;
            }
            
            return error;
        }
        
        void
        OptionParsingStarting ()
        {
            m_cascade = true;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        bool m_cascade;
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
public:
    CommandObjectTypeFormatAdd (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type format add",
                   "Add a new formatting style for a type.",
                   NULL), m_options (interpreter)
    {
        CommandArgumentEntry format_arg;
        CommandArgumentData format_style_arg;
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        format_style_arg.arg_type = eArgTypeFormat;
        format_style_arg.arg_repetition = eArgRepeatPlain;
                
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatPlus;
        
        format_arg.push_back (format_style_arg);
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (format_arg);
        m_arguments.push_back (type_arg);
    }
    
    ~CommandObjectTypeFormatAdd ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        if (argc < 2)
        {
            result.AppendErrorWithFormat ("%s takes two or more args.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        const char* format_cstr = command.GetArgumentAtIndex(0);
        
        if (!format_cstr || !format_cstr[0])
        {
            result.AppendError("empty format strings not allowed");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        lldb::Format format;
        Error error;
        
        error = Args::StringToFormat(format_cstr, format, NULL);
        ValueFormat::SharedPointer entry;
        
        entry.reset(new ValueFormat(format,m_options.m_cascade));

        if (error.Fail()) 
        {
            result.AppendError(error.AsCString());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        // now I have a valid format, let's add it to every type
        
        for(int i = 1; i < argc; i++) {
            const char* typeA = command.GetArgumentAtIndex(i);
            ConstString typeCS(typeA);
            if (typeCS)
                Debugger::ValueFormats::Add(typeCS, entry);
            else
            {
                result.AppendError("empty typenames not allowed");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }
        
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
        return result.Succeeded();
    }
        
};

OptionDefinition
CommandObjectTypeFormatAdd::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "cascade", 'c', required_argument, NULL, 0, eArgTypeBoolean,    "If true, cascade to derived typedefs."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectTypeFormatDelete
//-------------------------------------------------------------------------

class CommandObjectTypeFormatDelete : public CommandObject
{
public:
    CommandObjectTypeFormatDelete (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type format delete",
                   "Delete an existing formatting style for a type.",
                   NULL)
    {
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatPlain;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
        
    }
    
    ~CommandObjectTypeFormatDelete ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        if (argc != 1)
        {
            result.AppendErrorWithFormat ("%s takes 1 arg.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        const char* typeA = command.GetArgumentAtIndex(0);
        ConstString typeCS(typeA);
        
        if(!typeCS)
        {
            result.AppendError("empty typenames not allowed");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        
        if (Debugger::ValueFormats::Delete(typeCS))
        {
            result.SetStatus(eReturnStatusSuccessFinishNoResult);
            return result.Succeeded();
        }
        else
        {
            result.AppendErrorWithFormat ("no custom format for %s.\n", typeA);
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
    }
    
};

//-------------------------------------------------------------------------
// CommandObjectTypeFormatClear
//-------------------------------------------------------------------------

class CommandObjectTypeFormatClear : public CommandObject
{
public:
    CommandObjectTypeFormatClear (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type format clear",
                   "Delete all existing format styles.",
                   NULL)
    {
    }
    
    ~CommandObjectTypeFormatClear ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        Debugger::ValueFormats::Clear();
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
};

//-------------------------------------------------------------------------
// CommandObjectTypeFormatList
//-------------------------------------------------------------------------

bool CommandObjectTypeFormatList_LoopCallback(void* pt2self, const char* type, const ValueFormat::SharedPointer& entry);

class CommandObjectTypeFormatList;

struct CommandObjectTypeFormatList_LoopCallbackParam {
    CommandObjectTypeFormatList* self;
    CommandReturnObject* result;
    RegularExpression* regex;
    CommandObjectTypeFormatList_LoopCallbackParam(CommandObjectTypeFormatList* S, CommandReturnObject* R,
                                            RegularExpression* X = NULL) : self(S), result(R), regex(X) {}
};

class CommandObjectTypeFormatList : public CommandObject
{
public:
    CommandObjectTypeFormatList (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type format list",
                   "Show a list of current formatting styles.",
                   NULL)
    {
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatOptional;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
    }
    
    ~CommandObjectTypeFormatList ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        CommandObjectTypeFormatList_LoopCallbackParam *param;
        
        if (argc == 1) {
            RegularExpression* regex = new RegularExpression(command.GetArgumentAtIndex(0));
            regex->Compile(command.GetArgumentAtIndex(0));
            param = new CommandObjectTypeFormatList_LoopCallbackParam(this,&result,regex);
        }
        else
            param = new CommandObjectTypeFormatList_LoopCallbackParam(this,&result);
        Debugger::ValueFormats::LoopThrough(CommandObjectTypeFormatList_LoopCallback, param);
        delete param;
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
private:
    
    bool
    LoopCallback (const char* type,
                  const ValueFormat::SharedPointer& entry,
                  RegularExpression* regex,
                  CommandReturnObject *result)
    {
        if (regex == NULL || regex->Execute(type)) 
        {
            result->GetOutputStream().Printf ("%s: %s%s\n", type, 
                                              FormatManager::GetFormatAsCString (entry->m_format),
                                              entry->m_cascades ? "" : " (not cascading)");
        }
        return true;
    }
    
    friend bool CommandObjectTypeFormatList_LoopCallback(void* pt2self, const char* type, const ValueFormat::SharedPointer& entry);
    
};

bool
CommandObjectTypeFormatList_LoopCallback (
                                    void* pt2self,
                                    const char* type,
                                    const ValueFormat::SharedPointer& entry)
{
    CommandObjectTypeFormatList_LoopCallbackParam* param = (CommandObjectTypeFormatList_LoopCallbackParam*)pt2self;
    return param->self->LoopCallback(type, entry, param->regex, param->result);
}




//-------------------------------------------------------------------------
// CommandObjectTypeSummaryAdd
//-------------------------------------------------------------------------

class CommandObjectTypeSummaryAdd : public CommandObject
{
    
private:
    
    class CommandOptions : public Options
    {
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
        Options (interpreter)
        {
        }
        
        virtual
        ~CommandOptions (){}
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            bool success;
            
            switch (short_option)
            {
                case 'c':
                    m_cascade = Args::StringToBoolean(option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for cascade: %s.\n", option_arg);
                    break;
                case 'h':
                    m_no_children = !Args::StringToBoolean(option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for nochildren: %s.\n", option_arg);
                    break;
                case 'v':
                    m_no_value = !Args::StringToBoolean(option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for novalue: %s.\n", option_arg);
                    break;
                case 'o':
                    m_one_liner = Args::StringToBoolean(option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for oneliner: %s.\n", option_arg);
                    break;
                default:
                    error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
                    break;
            }
            
            return error;
        }
        
        void
        OptionParsingStarting ()
        {
            m_cascade = true;
            m_no_children = true;
            m_no_value = false;
            m_one_liner = false;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        bool m_cascade;
        bool m_no_children;
        bool m_no_value;
        bool m_one_liner;
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
public:
    CommandObjectTypeSummaryAdd (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type summary add",
                   "Add a new summary style for a type.",
                   NULL), m_options (interpreter)
    {
        CommandArgumentEntry format_arg;
        CommandArgumentData format_style_arg;
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        format_style_arg.arg_type = eArgTypeFormat;
        format_style_arg.arg_repetition = eArgRepeatPlain;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatPlus;
        
        format_arg.push_back (format_style_arg);
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (format_arg);
        m_arguments.push_back (type_arg);
    }
    
    ~CommandObjectTypeSummaryAdd ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        // we support just one custom syntax: type summary add -o yes typeName
        // anything else, must take the usual route
        // e.g. type summary add -o yes "" type1 type2 ... typeN
        
        bool isValidShortcut = m_options.m_one_liner && (argc == 1);
        bool isValid = (argc >= 2);
        
        if (!isValidShortcut && !isValid)
        {
            result.AppendErrorWithFormat ("%s takes two or more args.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        const char* format_cstr = (isValidShortcut ? "" : command.GetArgumentAtIndex(0));
        
        if ( (!format_cstr || !format_cstr[0]) && !m_options.m_one_liner )
        {
            result.AppendError("empty summary strings not allowed");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        Error error;
        
        SummaryFormat::SharedPointer entry(new SummaryFormat(format_cstr,m_options.m_cascade,
                                             m_options.m_no_children,m_options.m_no_value,
                                             m_options.m_one_liner));
        
        if (error.Fail()) 
        {
            result.AppendError(error.AsCString());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        // now I have a valid format, let's add it to every type
        
        for(int i = (isValidShortcut ? 0 : 1); i < argc; i++) {
            const char* typeA = command.GetArgumentAtIndex(i);
            ConstString typeCS(typeA);
            if (typeCS)
                Debugger::SummaryFormats::Add(typeCS, entry);
            else
            {
                result.AppendError("empty typenames not allowed");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
        return result.Succeeded();
    }
    
};

OptionDefinition
CommandObjectTypeSummaryAdd::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "cascade", 'c', required_argument, NULL, 0, eArgTypeBoolean,    "If true, cascade to derived typedefs."},
    { LLDB_OPT_SET_ALL, false, "show-children", 'h', required_argument, NULL, 0, eArgTypeBoolean,    "If true, print children."},
    { LLDB_OPT_SET_ALL, false, "show-value", 'v', required_argument, NULL, 0, eArgTypeBoolean,    "If true, print value."},
    { LLDB_OPT_SET_ALL, false, "one-liner", 'o', required_argument, NULL, 0, eArgTypeBoolean,    "If true, just print a one-line preformatted summary."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectTypeSummaryDelete
//-------------------------------------------------------------------------

class CommandObjectTypeSummaryDelete : public CommandObject
{
public:
    CommandObjectTypeSummaryDelete (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type summary delete",
                   "Delete an existing summary style for a type.",
                   NULL)
    {
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatPlain;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
        
    }
    
    ~CommandObjectTypeSummaryDelete ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        if (argc != 1)
        {
            result.AppendErrorWithFormat ("%s takes 1 arg.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        const char* typeA = command.GetArgumentAtIndex(0);
        ConstString typeCS(typeA);
        
        if(!typeCS)
        {
            result.AppendError("empty typenames not allowed");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        
        if (Debugger::SummaryFormats::Delete(typeCS))
        {
            result.SetStatus(eReturnStatusSuccessFinishNoResult);
            return result.Succeeded();
        }
        else
        {
            result.AppendErrorWithFormat ("no custom summary for %s.\n", typeA);
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
    }
    
};

//-------------------------------------------------------------------------
// CommandObjectTypeSummaryClear
//-------------------------------------------------------------------------

class CommandObjectTypeSummaryClear : public CommandObject
{
public:
    CommandObjectTypeSummaryClear (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type summary clear",
                   "Delete all existing summary styles.",
                   NULL)
    {
    }
    
    ~CommandObjectTypeSummaryClear ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        Debugger::SummaryFormats::Clear();
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
};

//-------------------------------------------------------------------------
// CommandObjectTypeSummaryList
//-------------------------------------------------------------------------

bool CommandObjectTypeSummaryList_LoopCallback(void* pt2self, const char* type, const SummaryFormat::SharedPointer& entry);

class CommandObjectTypeSummaryList;

struct CommandObjectTypeSummaryList_LoopCallbackParam {
    CommandObjectTypeSummaryList* self;
    CommandReturnObject* result;
    RegularExpression* regex;
    CommandObjectTypeSummaryList_LoopCallbackParam(CommandObjectTypeSummaryList* S, CommandReturnObject* R,
                                                  RegularExpression* X = NULL) : self(S), result(R), regex(X) {}
};

class CommandObjectTypeSummaryList : public CommandObject
{
public:
    CommandObjectTypeSummaryList (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type summary list",
                   "Show a list of current summary styles.",
                   NULL)
    {
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatOptional;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
    }
    
    ~CommandObjectTypeSummaryList ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        CommandObjectTypeSummaryList_LoopCallbackParam *param;
        
        if (argc == 1) {
            RegularExpression* regex = new RegularExpression(command.GetArgumentAtIndex(0));
            regex->Compile(command.GetArgumentAtIndex(0));
            param = new CommandObjectTypeSummaryList_LoopCallbackParam(this,&result,regex);
        }
        else
            param = new CommandObjectTypeSummaryList_LoopCallbackParam(this,&result);
        Debugger::SummaryFormats::LoopThrough(CommandObjectTypeSummaryList_LoopCallback, param);
        delete param;
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
private:
    
    bool
    LoopCallback (const char* type,
                  const SummaryFormat::SharedPointer& entry,
                  RegularExpression* regex,
                  CommandReturnObject *result)
    {
        if (regex == NULL || regex->Execute(type)) 
        {
                result->GetOutputStream().Printf ("%s: `%s`%s%s%s%s\n", type, 
                                                  entry->m_format.c_str(),
                                                  entry->m_cascades ? "" : " (not cascading)",
                                                  entry->m_dont_show_children ? "" : " (show children)",
                                                  entry->m_dont_show_value ? "" : " (show value)",
                                                  entry->m_show_members_oneliner ? " (one-line printout)" : "");
        }
        return true;
    }
    
    friend bool CommandObjectTypeSummaryList_LoopCallback(void* pt2self, const char* type, const SummaryFormat::SharedPointer& entry);
    
};

bool
CommandObjectTypeSummaryList_LoopCallback (
                                          void* pt2self,
                                          const char* type,
                                          const SummaryFormat::SharedPointer& entry)
{
    CommandObjectTypeSummaryList_LoopCallbackParam* param = (CommandObjectTypeSummaryList_LoopCallbackParam*)pt2self;
    return param->self->LoopCallback(type, entry, param->regex, param->result);
}





class CommandObjectTypeFormat : public CommandObjectMultiword
{
public:
    CommandObjectTypeFormat (CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter,
                                "type format",
                                "A set of commands for editing variable value display options",
                                "type format [<sub-command-options>] ")
    {
        LoadSubCommand ("add",    CommandObjectSP (new CommandObjectTypeFormatAdd (interpreter)));
        LoadSubCommand ("clear",  CommandObjectSP (new CommandObjectTypeFormatClear (interpreter)));
        LoadSubCommand ("delete", CommandObjectSP (new CommandObjectTypeFormatDelete (interpreter)));
        LoadSubCommand ("list",   CommandObjectSP (new CommandObjectTypeFormatList (interpreter)));
    }


    ~CommandObjectTypeFormat ()
    {
    }
};

class CommandObjectTypeSummary : public CommandObjectMultiword
{
public:
    CommandObjectTypeSummary (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "type format",
                            "A set of commands for editing variable summary display options",
                            "type summary [<sub-command-options>] ")
    {
        LoadSubCommand ("add",    CommandObjectSP (new CommandObjectTypeSummaryAdd (interpreter)));
        LoadSubCommand ("clear",  CommandObjectSP (new CommandObjectTypeSummaryClear (interpreter)));
        LoadSubCommand ("delete", CommandObjectSP (new CommandObjectTypeSummaryDelete (interpreter)));
        LoadSubCommand ("list",   CommandObjectSP (new CommandObjectTypeSummaryList (interpreter)));
    }
    
    
    ~CommandObjectTypeSummary ()
    {
    }
};

//-------------------------------------------------------------------------
// CommandObjectType
//-------------------------------------------------------------------------

CommandObjectType::CommandObjectType (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "type",
                            "A set of commands for operating on the type system",
                            "type [<sub-command-options>]")
{
    LoadSubCommand ("format",    CommandObjectSP (new CommandObjectTypeFormat (interpreter)));
    LoadSubCommand ("summary",   CommandObjectSP (new CommandObjectTypeSummary (interpreter)));
}


CommandObjectType::~CommandObjectType ()
{
}


