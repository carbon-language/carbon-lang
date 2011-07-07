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
#include "lldb/Core/RegularExpression.h"
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
                case 'C':
                    m_cascade = Args::StringToBoolean(option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for cascade: %s.\n", option_arg);
                    break;
                case 'f':
                    error = Args::StringToFormat(option_arg, m_format, NULL);
                    break;
                case 'p':
                    m_skip_pointers = true;
                    break;
                case 'r':
                    m_skip_references = true;
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
            m_format = eFormatInvalid;
            m_skip_pointers = false;
            m_skip_references = false;
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
        lldb::Format m_format;
        bool m_skip_references;
        bool m_skip_pointers;
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
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatPlus;
        
        type_arg.push_back (type_style_arg);

        m_arguments.push_back (type_arg);
        
        SetHelpLong(
                    "Some examples of using this command.\n"
                    "We use as reference the following snippet of code:\n"
                    "\n"
                    "typedef int Aint;\n"
                    "typedef float Afloat;\n"
                    "typedef Aint Bint;\n"
                    "typedef Afloat Bfloat;\n"
                    "\n"
                    "Aint ix = 5;\n"
                    "Bint iy = 5;\n"
                    "\n"
                    "Afloat fx = 3.14;\n"
                    "BFloat fy = 3.14;\n"
                    "\n"
                    "Typing:\n"
                    "type format add -f hex AInt\n"
                    "frame variable iy\n"
                    "will produce an hex display of iy, because no formatter is available for Bint and the one for Aint is used instead\n"
                    "To prevent this type\n"
                    "type format add -f hex -C no AInt\n"
                    "\n"
                    "A similar reasoning applies to\n"
                    "type format add -f hex -C no float -p\n"
                    "which now prints all floats and float&s as hexadecimal, but does not format float*s\n"
                    "and does not change the default display for Afloat and Bfloat objects.\n"
                    );
    }
    
    ~CommandObjectTypeFormatAdd ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        if (argc < 1)
        {
            result.AppendErrorWithFormat ("%s takes one or more args.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        if(m_options.m_format == eFormatInvalid)
        {
            result.AppendErrorWithFormat ("%s needs a valid format.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        ValueFormatSP entry;
        
        entry.reset(new ValueFormat(m_options.m_format,
                                    m_options.m_cascade,
                                    m_options.m_skip_pointers,
                                    m_options.m_skip_references));

        // now I have a valid format, let's add it to every type
        
        for(int i = 0; i < argc; i++) {
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
    { LLDB_OPT_SET_ALL, false, "cascade", 'C', required_argument, NULL, 0, eArgTypeBoolean,    "If true, cascade to derived typedefs."},
    { LLDB_OPT_SET_ALL, false, "format", 'f', required_argument, NULL, 0, eArgTypeFormat,    "The format to use to display this type."},
    { LLDB_OPT_SET_ALL, false, "skip-pointers", 'p', no_argument, NULL, 0, eArgTypeBoolean,         "Don't use this format for pointers-to-type objects."},
    { LLDB_OPT_SET_ALL, false, "skip-references", 'r', no_argument, NULL, 0, eArgTypeBoolean,         "Don't use this format for references-to-type objects."},
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
            result->GetOutputStream().Printf ("%s: %s%s%s%s\n", type, 
                                              FormatManager::GetFormatAsCString (entry->m_format),
                                              entry->m_cascades ? "" : " (not cascading)",
                                              entry->m_skip_pointers ? " (skip pointers)" : "",
                                              entry->m_skip_references ? " (skip references)" : "");
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
                case 'C':
                    m_cascade = Args::StringToBoolean(option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for cascade: %s.\n", option_arg);
                    break;
                case 'e':
                    m_no_children = false;
                    break;
                case 'v':
                    m_no_value = true;
                    break;
                case 'c':
                    m_one_liner = true;
                    break;
                case 'f':
                    m_format_string = std::string(option_arg);
                    break;
                case 'p':
                    m_skip_pointers = true;
                    break;
                case 'r':
                    m_skip_references = true;
                    break;
                case 'x':
                    m_regex = true;
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
            m_skip_references = false;
            m_skip_pointers = false;
            m_regex = false;
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
        bool m_skip_references;
        bool m_skip_pointers;
        bool m_regex;
        std::string m_format_string;
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
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatPlus;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
        
        SetHelpLong(
                    "Some examples of using this command.\n"
                    "We use as reference the following snippet of code:\n"
                    "struct JustADemo\n"
                    "{\n"
                    "int* ptr;\n"
                    "float value;\n"
                    "JustADemo(int p = 1, float v = 0.1) : ptr(new int(p)), value(v) {}\n"
                    "};\n"
                    "JustADemo object(42,3.14);\n"
                    "struct AnotherDemo : public JustADemo\n"
                    "{\n"
                    "uint8_t byte;\n"
                    "AnotherDemo(uint8_t b = 'E', int p = 1, float v = 0.1) : JustADemo(p,v), byte(b) {}\n"
                    "};\n"
                    "AnotherDemo *another_object = new AnotherDemo('E',42,3.14);\n"
                    "\n"
                    "type summary add -f \"the answer is ${*var.ptr}\" JustADemo\n"
                    "when typing frame variable object you will get \"the answer is 42\"\n"
                    "type summary add -f \"the answer is ${*var.ptr}, and the question is ${var.value}\" JustADemo\n"
                    "when typing frame variable object you will get \"the answer is 42 and the question is 3.14\"\n"
                    "\n"
                    "Alternatively, you could also say\n"
                    "type summary add -f \"${var%V} -> ${*var}\" \"int *\"\n"
                    "and replace the above summary string with\n"
                    "type summary add -f \"the answer is ${var.ptr}, and the question is ${var.value}\" JustADemo\n"
                    "to obtain a similar result\n"
                    "\n"
                    "To add a summary valid for both JustADemo and AnotherDemo you can use the scoping operator, as in:\n"
                    "type summary add -f \"${var.ptr}, ${var.value},{${var.byte}}\" JustADemo -C yes\n"
                    "\n"
                    "This will be used for both variables of type JustADemo and AnotherDemo. To prevent this, change the -C to read -C no\n"
                    "If you do not want pointers to be shown using that summary, you can use the -p option, as in:\n"
                    "type summary add -f \"${var.ptr}, ${var.value},{${var.byte}}\" JustADemo -C yes -p\n"
                    "A similar option -r exists for references.\n"
                    "\n"
                    "If you simply want a one-line summary of the content of your variable, without typing an explicit string to that effect\n"
                    "you can use the -c option, without giving any summary string:\n"
                    "type summary add -c JustADemo\n"
                    "frame variable object\n"
                    "the output being similar to (ptr=0xsomeaddress, value=3.14)\n"
                    "\n"
                    "If you want to display some summary text, but also expand the structure of your object, you can add the -e option, as in:\n"
                    "type summary add -e -f \"*ptr = ${*var.ptr}\" JustADemo\n"
                    "Here the value of the int* is displayed, followed by the standard LLDB sequence of children objects, one per line.\n"
                    "to get an output like:\n"
                    "\n"
                    "*ptr = 42 {\n"
                    " ptr = 0xsomeaddress\n"
                    " value = 3.14\n"
                    "}\n"
                    "\n"
                    "A command you may definitely want to try if you're doing C++ debugging is:\n"
                    "type summary add -f \"${var._M_dataplus._M_p}\" std::string\n"
        );
    }
    
    ~CommandObjectTypeSummaryAdd ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        if (argc < 1)
        {
            result.AppendErrorWithFormat ("%s takes one or more args.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        if(!m_options.m_one_liner && m_options.m_format_string.empty())
        {
            result.AppendError("empty summary strings not allowed");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        const char* format_cstr = (m_options.m_one_liner ? "" : m_options.m_format_string.c_str());
        
        Error error;
        
        SummaryFormat::SharedPointer entry(new SummaryFormat(format_cstr,m_options.m_cascade,
                                             m_options.m_no_children,m_options.m_no_value,
                                             m_options.m_one_liner,
                                             m_options.m_skip_pointers,
                                             m_options.m_skip_references));
        
        if (error.Fail()) 
        {
            result.AppendError(error.AsCString());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        // now I have a valid format, let's add it to every type
        
        for(int i = 0; i < argc; i++) {
            const char* typeA = command.GetArgumentAtIndex(i);
            if(!typeA || typeA[0] == '\0')
            {
                result.AppendError("empty typenames not allowed");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            ConstString typeCS(typeA);
            if(!m_options.m_regex)
            {
                Debugger::SummaryFormats::Add(typeCS, entry);
            }
            else
            {
                RegularExpressionSP typeRX(new RegularExpression());
                if(!typeRX->Compile(typeA))
                {
                    result.AppendError("regex format error (maybe this is not really a regex?)");
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                Debugger::RegexSummaryFormats::Delete(typeCS);
                Debugger::RegexSummaryFormats::Add(typeRX, entry);
            }
        }
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
        return result.Succeeded();
    }
    
};

OptionDefinition
CommandObjectTypeSummaryAdd::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "cascade", 'C', required_argument, NULL, 0, eArgTypeBoolean,    "If true, cascade to derived typedefs."},
    { LLDB_OPT_SET_ALL, false, "no-value", 'v', no_argument, NULL, 0, eArgTypeBoolean,         "Don't show the value, just show the summary, for this type."},
    { LLDB_OPT_SET_ALL, false, "skip-pointers", 'p', no_argument, NULL, 0, eArgTypeBoolean,         "Don't use this format for pointers-to-type objects."},
    { LLDB_OPT_SET_ALL, false, "skip-references", 'r', no_argument, NULL, 0, eArgTypeBoolean,         "Don't use this format for references-to-type objects."},
    { LLDB_OPT_SET_ALL, false,  "regex", 'x', no_argument, NULL, 0, eArgTypeBoolean,    "Type names are actually regular expressions."},
    { LLDB_OPT_SET_1  , true, "inline-children", 'c', no_argument, NULL, 0, eArgTypeBoolean,    "If true, inline all child values into summary string."},
    { LLDB_OPT_SET_2  , true, "format-string", 'f', required_argument, NULL, 0, eArgTypeSummaryString,    "Format string used to display text and object contents."},
    { LLDB_OPT_SET_2, false, "expand", 'e', no_argument, NULL, 0, eArgTypeBoolean,    "Expand aggregate data types to show children on separate lines."},
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
        
        bool delete_summary = Debugger::SummaryFormats::Delete(typeCS);
        bool delete_regex = Debugger::RegexSummaryFormats::Delete(typeCS);
        if (delete_summary || delete_regex)
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
        Debugger::RegexSummaryFormats::Clear();
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
};

//-------------------------------------------------------------------------
// CommandObjectTypeSummaryList
//-------------------------------------------------------------------------

bool CommandObjectTypeSummaryList_LoopCallback(void* pt2self, const char* type, const SummaryFormat::SharedPointer& entry);
bool CommandObjectTypeRXSummaryList_LoopCallback(void* pt2self, lldb::RegularExpressionSP regex, const SummaryFormat::SharedPointer& entry);

class CommandObjectTypeSummaryList;

struct CommandObjectTypeSummaryList_LoopCallbackParam {
    CommandObjectTypeSummaryList* self;
    CommandReturnObject* result;
    RegularExpression* regex;
    CommandObjectTypeSummaryList_LoopCallbackParam(CommandObjectTypeSummaryList* S, CommandReturnObject* R,
                                                  RegularExpression* X = NULL) : self(S), result(R), regex(X) {}
};

struct CommandObjectTypeRXSummaryList_LoopCallbackParam {
    CommandObjectTypeSummaryList* self;
    CommandReturnObject* result;
    RegularExpression* regex;
    CommandObjectTypeRXSummaryList_LoopCallbackParam(CommandObjectTypeSummaryList* S, CommandReturnObject* R,
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
        CommandObjectTypeRXSummaryList_LoopCallbackParam *rxparam;
        
        if (argc == 1) {
            RegularExpression* regex = new RegularExpression(command.GetArgumentAtIndex(0));
            regex->Compile(command.GetArgumentAtIndex(0));
            param = new CommandObjectTypeSummaryList_LoopCallbackParam(this,&result,regex);
        }
        else
            param = new CommandObjectTypeSummaryList_LoopCallbackParam(this,&result);
        Debugger::SummaryFormats::LoopThrough(CommandObjectTypeSummaryList_LoopCallback, param);
        delete param;
        
        if(Debugger::RegexSummaryFormats::GetCount() == 0)
        {
            result.SetStatus(eReturnStatusSuccessFinishResult);
            return result.Succeeded();
        }
        
        result.GetOutputStream().Printf("Regex-based summaries (slower):\n");
        if (argc == 1) {
            RegularExpression* regex = new RegularExpression(command.GetArgumentAtIndex(0));
            regex->Compile(command.GetArgumentAtIndex(0));
            rxparam = new CommandObjectTypeRXSummaryList_LoopCallbackParam(this,&result,regex);
        }
        else
            rxparam = new CommandObjectTypeRXSummaryList_LoopCallbackParam(this,&result);
        Debugger::RegexSummaryFormats::LoopThrough(CommandObjectTypeRXSummaryList_LoopCallback, rxparam);
        delete rxparam;
        
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
                result->GetOutputStream().Printf ("%s: `%s`%s%s%s%s%s%s\n", type, 
                                                  entry->m_format.c_str(),
                                                  entry->m_cascades ? "" : " (not cascading)",
                                                  entry->m_dont_show_children ? "" : " (show children)",
                                                  entry->m_dont_show_value ? " (hide value)" : "",
                                                  entry->m_show_members_oneliner ? " (one-line printout)" : "",
                                                  entry->m_skip_pointers ? " (skip pointers)" : "",
                                                  entry->m_skip_references ? " (skip references)" : "");
        }
        return true;
    }
    
    friend bool CommandObjectTypeSummaryList_LoopCallback(void* pt2self, const char* type, const SummaryFormat::SharedPointer& entry);
    friend bool CommandObjectTypeRXSummaryList_LoopCallback(void* pt2self, lldb::RegularExpressionSP regex, const SummaryFormat::SharedPointer& entry);

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

bool
CommandObjectTypeRXSummaryList_LoopCallback (
                                           void* pt2self,
                                           lldb::RegularExpressionSP regex,
                                           const SummaryFormat::SharedPointer& entry)
{
    CommandObjectTypeRXSummaryList_LoopCallbackParam* param = (CommandObjectTypeRXSummaryList_LoopCallbackParam*)pt2self;
    return param->self->LoopCallback(regex->GetText(), entry, param->regex, param->result);
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
                            "type summary",
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


