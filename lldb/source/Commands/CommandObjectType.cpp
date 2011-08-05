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

#include <ctype.h>

// C++ Includes

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/FormatManager.h"
#include "lldb/Core/InputReaderEZ.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StringList.h"
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
        
        if (m_options.m_format == eFormatInvalid)
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
        
        for (size_t i = 0; i < argc; i++) {
            const char* typeA = command.GetArgumentAtIndex(i);
            ConstString typeCS(typeA);
            if (typeCS)
                Debugger::Formatting::ValueFormats::Add(typeCS, entry);
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
    { LLDB_OPT_SET_ALL, false, "skip-pointers", 'p', no_argument, NULL, 0, eArgTypeNone,         "Don't use this format for pointers-to-type objects."},
    { LLDB_OPT_SET_ALL, false, "skip-references", 'r', no_argument, NULL, 0, eArgTypeNone,         "Don't use this format for references-to-type objects."},
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
        
        if (!typeCS)
        {
            result.AppendError("empty typenames not allowed");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        
        if (Debugger::Formatting::ValueFormats::Delete(typeCS))
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
        Debugger::Formatting::ValueFormats::Clear();
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
        Debugger::Formatting::ValueFormats::LoopThrough(CommandObjectTypeFormatList_LoopCallback, param);
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

static const char *g_summary_addreader_instructions = "Enter your Python command(s). Type 'DONE' to end.\n"
                                                       "def function (valobj,dict):";

class TypeScriptAddInputReader : public InputReaderEZ
{
private:
    DISALLOW_COPY_AND_ASSIGN (TypeScriptAddInputReader);
public:
    TypeScriptAddInputReader(Debugger& debugger) : 
    InputReaderEZ(debugger)
    {}
    
    virtual
    ~TypeScriptAddInputReader()
    {
    }
    
    virtual void ActivateHandler(HandlerData& data)
    {
        StreamSP out_stream = data.reader.GetDebugger().GetAsyncOutputStream();
        bool batch_mode = data.reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
        if (!batch_mode)
        {
            out_stream->Printf ("%s\n", g_summary_addreader_instructions);
            if (data.reader.GetPrompt())
                out_stream->Printf ("%s", data.reader.GetPrompt());
            out_stream->Flush();
        }
    }
    
    virtual void ReactivateHandler(HandlerData& data)
    {
        StreamSP out_stream = data.reader.GetDebugger().GetAsyncOutputStream();
        bool batch_mode = data.reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
        if (data.reader.GetPrompt() && !batch_mode)
        {
            out_stream->Printf ("%s", data.reader.GetPrompt());
            out_stream->Flush();
        }
    }
    virtual void GotTokenHandler(HandlerData& data)
    {
        StreamSP out_stream = data.reader.GetDebugger().GetAsyncOutputStream();
        bool batch_mode = data.reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
        if (data.bytes && data.bytes_len && data.baton)
        {
            ((ScriptAddOptions*)data.baton)->m_user_source.AppendString(data.bytes, data.bytes_len);
        }
        if (!data.reader.IsDone() && data.reader.GetPrompt() && !batch_mode)
        {
            out_stream->Printf ("%s", data.reader.GetPrompt());
            out_stream->Flush();
        }
    }
    virtual void InterruptHandler(HandlerData& data)
    {
        StreamSP out_stream = data.reader.GetDebugger().GetAsyncOutputStream();
        bool batch_mode = data.reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
        data.reader.SetIsDone (true);
        if (!batch_mode)
        {
            out_stream->Printf ("Warning: No command attached to breakpoint.\n");
            out_stream->Flush();
        }
    }
    virtual void EOFHandler(HandlerData& data)
    {
        data.reader.SetIsDone (true);
    }
    virtual void DoneHandler(HandlerData& data)
    {
        StreamSP out_stream = data.reader.GetDebugger().GetAsyncOutputStream();
        ScriptAddOptions *options_ptr = ((ScriptAddOptions*)data.baton);
        if (!options_ptr)
        {
            out_stream->Printf ("Internal error #1: no script attached.\n");
            out_stream->Flush();
            return;
        }
        
        ScriptAddOptions::SharedPointer options(options_ptr); // this will ensure that we get rid of the pointer when going out of scope
        
        ScriptInterpreter *interpreter = data.reader.GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
        if (!interpreter)
        {
            out_stream->Printf ("Internal error #2: no script attached.\n");
            out_stream->Flush();
            return;
        }
        StringList funct_name_sl;
        if (!interpreter->GenerateTypeScriptFunction (options->m_user_source, 
                                                      funct_name_sl))
        {
            out_stream->Printf ("Internal error #3: no script attached.\n");
            out_stream->Flush();
            return;
        }
        if (funct_name_sl.GetSize() == 0)
        {
            out_stream->Printf ("Internal error #4: no script attached.\n");
            out_stream->Flush();
            return;
        }
        const char *funct_name = funct_name_sl.GetStringAtIndex(0);
        if (!funct_name || !funct_name[0])
        {
            out_stream->Printf ("Internal error #5: no script attached.\n");
            out_stream->Flush();
            return;
        }
        // now I have a valid function name, let's add this as script for every type in the list
        
        SummaryFormatSP script_format;
        script_format.reset(new ScriptSummaryFormat(options->m_cascade,
                                                    options->m_skip_pointers,
                                                    options->m_skip_references,
                                                    options->m_no_children,
                                                    options->m_no_value,
                                                    options->m_one_liner,
                                                    std::string(funct_name),
                                                    options->m_user_source.CopyList("     ")));
        
        Error error;
        
        for (size_t i = 0; i < options->m_target_types.GetSize(); i++)
        {
            const char *type_name = options->m_target_types.GetStringAtIndex(i);
            CommandObjectTypeSummaryAdd::AddSummary(ConstString(type_name),
                                                    script_format,
                                                    (options->m_regex ? CommandObjectTypeSummaryAdd::eRegexSummary : CommandObjectTypeSummaryAdd::eRegularSummary),
                                                    options->m_category,
                                                    &error);
            if (error.Fail())
            {
                out_stream->Printf (error.AsCString());
                out_stream->Flush();
                return;
            }
        }
        
        if (options->m_name)
        {
            if ( (bool)(*(options->m_name)) )
            {
                CommandObjectTypeSummaryAdd::AddSummary(*(options->m_name),
                                                                  script_format,
                                                                  CommandObjectTypeSummaryAdd::eNamedSummary,
                                                                  options->m_category,
                                                                  &error);
                if (error.Fail())
                {
                    out_stream->Printf (error.AsCString());
                    out_stream->Flush();
                    return;
                }
            }
            else
            {
                out_stream->Printf (error.AsCString());
                out_stream->Flush();
                return;
            }
        }
    }
};

Error
CommandObjectTypeSummaryAdd::CommandOptions::SetOptionValue (uint32_t option_idx, const char *option_arg)
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
        case 'n':
            m_name = new ConstString(option_arg);
            break;
        case 's':
            m_python_script = std::string(option_arg);
            m_is_add_script = true;
            break;
        case 'F':
            m_python_function = std::string(option_arg);
            m_is_add_script = true;
            break;
        case 'P':
            m_is_add_script = true;
            break;
        case 'w':
            m_category = ConstString(option_arg).GetCString();
            break;
        default:
            error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
            break;
    }
    
    return error;
}

void
CommandObjectTypeSummaryAdd::CommandOptions::OptionParsingStarting ()
{
    m_cascade = true;
    m_no_children = true;
    m_no_value = false;
    m_one_liner = false;
    m_skip_references = false;
    m_skip_pointers = false;
    m_regex = false;
    m_name = NULL;
    m_python_script = "";
    m_python_function = "";
    m_is_add_script = false;
    m_category = NULL;
}

void
CommandObjectTypeSummaryAdd::CollectPythonScript (ScriptAddOptions *options,
                                                  CommandReturnObject &result)
{
    InputReaderSP reader_sp (new TypeScriptAddInputReader(m_interpreter.GetDebugger()));
    if (reader_sp && options)
    {
        
        InputReaderEZ::InitializationParameters ipr;
        
        Error err (reader_sp->Initialize (ipr.SetBaton(options).SetPrompt("     ")));
        if (err.Success())
        {
            m_interpreter.GetDebugger().PushInputReader (reader_sp);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError (err.AsCString());
            result.SetStatus (eReturnStatusFailed);
        }
    }
    else
    {
        result.AppendError("out of memory");
        result.SetStatus (eReturnStatusFailed);
    }
}

bool
CommandObjectTypeSummaryAdd::Execute_ScriptSummary (Args& command, CommandReturnObject &result)
{
    const size_t argc = command.GetArgumentCount();
    
    if (argc < 1 && !m_options.m_name)
    {
        result.AppendErrorWithFormat ("%s takes one or more args.\n", m_cmd_name.c_str());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
    
    SummaryFormatSP script_format;
    
    if (!m_options.m_python_function.empty()) // we have a Python function ready to use
    {
        ScriptInterpreter *interpreter = m_interpreter.GetScriptInterpreter();
        if (!interpreter)
        {
            result.AppendError ("Internal error #1N: no script attached.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        const char *funct_name = m_options.m_python_function.c_str();
        if (!funct_name || !funct_name[0])
        {
            result.AppendError ("Internal error #2N: no script attached.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        script_format.reset(new ScriptSummaryFormat(m_options.m_cascade,
                                                    m_options.m_skip_pointers,
                                                    m_options.m_skip_references,
                                                    m_options.m_no_children,
                                                    m_options.m_no_value,
                                                    m_options.m_one_liner,
                                                    std::string(funct_name),
                                                    "     " + m_options.m_python_function + "(valobj,dict)"));
    }
    else if (!m_options.m_python_script.empty()) // we have a quick 1-line script, just use it
    {
        ScriptInterpreter *interpreter = m_interpreter.GetScriptInterpreter();
        if (!interpreter)
        {
            result.AppendError ("Internal error #1Q: no script attached.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        StringList funct_sl;
        funct_sl << m_options.m_python_script.c_str();
        StringList funct_name_sl;
        if (!interpreter->GenerateTypeScriptFunction (funct_sl, 
                                                      funct_name_sl))
        {
            result.AppendError ("Internal error #2Q: no script attached.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        if (funct_name_sl.GetSize() == 0)
        {
            result.AppendError ("Internal error #3Q: no script attached.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        const char *funct_name = funct_name_sl.GetStringAtIndex(0);
        if (!funct_name || !funct_name[0])
        {
            result.AppendError ("Internal error #4Q: no script attached.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        script_format.reset(new ScriptSummaryFormat(m_options.m_cascade,
                                                    m_options.m_skip_pointers,
                                                    m_options.m_skip_references,
                                                    m_options.m_no_children,
                                                    m_options.m_no_value,
                                                    m_options.m_one_liner,
                                                    std::string(funct_name),
                                                    "     " + m_options.m_python_script));
    }
    else // use an InputReader to grab Python code from the user
    {        
        ScriptAddOptions *options = new ScriptAddOptions(m_options.m_skip_pointers,
                                                         m_options.m_skip_references,
                                                         m_options.m_cascade,
                                                         m_options.m_no_children,
                                                         m_options.m_no_value,
                                                         m_options.m_one_liner,
                                                         m_options.m_regex,
                                                         m_options.m_name,
                                                         m_options.m_category);
        
        for (size_t i = 0; i < argc; i++) {
            const char* typeA = command.GetArgumentAtIndex(i);
            if (typeA && *typeA)
                options->m_target_types << typeA;
            else
            {
                result.AppendError("empty typenames not allowed");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }
        
        CollectPythonScript(options,result);
        return result.Succeeded();
    }
    
    // if I am here, script_format must point to something good, so I can add that
    // as a script summary to all interested parties
    
    Error error;
    
    for (size_t i = 0; i < command.GetArgumentCount(); i++)
    {
        const char *type_name = command.GetArgumentAtIndex(i);
        CommandObjectTypeSummaryAdd::AddSummary(ConstString(type_name),
                                                script_format,
                                                (m_options.m_regex ? eRegexSummary : eRegularSummary),
                                                m_options.m_category,
                                                &error);
        if (error.Fail())
        {
            result.AppendError(error.AsCString());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
    }
    
    if (m_options.m_name)
    {
        if ( (bool)(*(m_options.m_name)) )
        {
            AddSummary(*(m_options.m_name), script_format, eNamedSummary, m_options.m_category, &error);
            if (error.Fail())
            {
                result.AppendError(error.AsCString());
                result.AppendError("added to types, but not given a name");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }
        else
        {
            result.AppendError("added to types, but not given a name");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
    }
    
    return result.Succeeded();
}

bool
CommandObjectTypeSummaryAdd::Execute_StringSummary (Args& command, CommandReturnObject &result)
{
    const size_t argc = command.GetArgumentCount();
    
    if (argc < 1 && !m_options.m_name)
    {
        result.AppendErrorWithFormat ("%s takes one or more args.\n", m_cmd_name.c_str());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
    
    if (!m_options.m_one_liner && m_options.m_format_string.empty())
    {
        result.AppendError("empty summary strings not allowed");
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
    
    const char* format_cstr = (m_options.m_one_liner ? "" : m_options.m_format_string.c_str());
    
    Error error;
    
    SummaryFormat::SharedPointer entry(new StringSummaryFormat(m_options.m_cascade,
                                                               m_options.m_skip_pointers,
                                                               m_options.m_skip_references,
                                                               m_options.m_no_children,
                                                               m_options.m_no_value,
                                                               m_options.m_one_liner,
                                                               format_cstr));
    
    if (error.Fail()) 
    {
        result.AppendError(error.AsCString());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
    
    // now I have a valid format, let's add it to every type
    
    for (size_t i = 0; i < argc; i++) {
        const char* typeA = command.GetArgumentAtIndex(i);
        if (!typeA || typeA[0] == '\0')
        {
            result.AppendError("empty typenames not allowed");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        ConstString typeCS(typeA);
        
        AddSummary(typeCS,
                   entry,
                   (m_options.m_regex ? eRegexSummary : eRegularSummary),
                   m_options.m_category,
                   &error);
        
        if (error.Fail())
        {
            result.AppendError(error.AsCString());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
    }
    
    if (m_options.m_name)
    {
        if ( (bool)(*(m_options.m_name)) )
        {
            AddSummary(*(m_options.m_name), entry, eNamedSummary, m_options.m_category, &error);
            if (error.Fail())
            {
                result.AppendError(error.AsCString());
                result.AppendError("added to types, but not given a name");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }
        else
        {
            result.AppendError("added to types, but not given a name");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
    }
    
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return result.Succeeded();
}

CommandObjectTypeSummaryAdd::CommandObjectTypeSummaryAdd (CommandInterpreter &interpreter) :
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
                "\n"
                "You can also add Python summaries, in which case you will use lldb public API to gather information from your variables"
                "and elaborate them to a meaningful summary inside a script written in Python. The variable object will be passed to your"
                "script as an SBValue object. The following example might help you when starting to use the Python summaries feature:\n"
                "type summary add JustADemo -s \"value = valobj.GetChildMemberWithName('value'); return 'My value is ' + value.GetValue();\"\n"
                "If you prefer to type your scripts on multiple lines, you will use the -P option and then type your script, ending it with "
                "the word DONE on a line by itself to mark you're finished editing your code:\n"
                "(lldb)type summary add JustADemo -P\n"
                "     value = valobj.GetChildMemberWithName('value');\n"
                "     return 'My value is ' + value.GetValue();\n"
                "DONE\n"
                "(lldb) <-- type further LLDB commands here\n"
                );
}

bool
CommandObjectTypeSummaryAdd::Execute (Args& command, CommandReturnObject &result)
{
    if (m_options.m_is_add_script)
        return Execute_ScriptSummary(command, result);
    else
        return Execute_StringSummary(command, result);
}

bool
CommandObjectTypeSummaryAdd::AddSummary(const ConstString& type_name,
                                        SummaryFormatSP entry,
                                        SummaryFormatType type,
                                        const char* category_name,
                                        Error* error)
{
    lldb::FormatCategorySP category;
    Debugger::Formatting::Categories::Get(ConstString(category_name), category);
    
    if (type == eRegexSummary)
    {
        RegularExpressionSP typeRX(new RegularExpression());
        if (!typeRX->Compile(type_name.GetCString()))
        {
            if (error)
                error->SetErrorString("regex format error (maybe this is not really a regex?)");
            return false;
        }
        
        category->RegexSummary()->Delete(type_name.GetCString());
        category->RegexSummary()->Add(typeRX, entry);
        
        return true;
    }
    else if (type == eNamedSummary)
    {
        // system named summaries do not exist (yet?)
        Debugger::Formatting::NamedSummaryFormats::Add(type_name,entry);
        return true;
    }
    else
    {
        category->Summary()->Add(type_name.GetCString(), entry);
        return true;
    }
}    

OptionDefinition
CommandObjectTypeSummaryAdd::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "category", 'w', required_argument, NULL, 0, eArgTypeName,    "Add this to the given category instead of the default one."},
    { LLDB_OPT_SET_ALL, false, "cascade", 'C', required_argument, NULL, 0, eArgTypeBoolean,    "If true, cascade to derived typedefs."},
    { LLDB_OPT_SET_ALL, false, "no-value", 'v', no_argument, NULL, 0, eArgTypeNone,         "Don't show the value, just show the summary, for this type."},
    { LLDB_OPT_SET_ALL, false, "skip-pointers", 'p', no_argument, NULL, 0, eArgTypeNone,         "Don't use this format for pointers-to-type objects."},
    { LLDB_OPT_SET_ALL, false, "skip-references", 'r', no_argument, NULL, 0, eArgTypeNone,         "Don't use this format for references-to-type objects."},
    { LLDB_OPT_SET_ALL, false,  "regex", 'x', no_argument, NULL, 0, eArgTypeNone,    "Type names are actually regular expressions."},
    { LLDB_OPT_SET_1  , true, "inline-children", 'c', no_argument, NULL, 0, eArgTypeNone,    "If true, inline all child values into summary string."},
    { LLDB_OPT_SET_2  , true, "format-string", 'f', required_argument, NULL, 0, eArgTypeSummaryString,    "Format string used to display text and object contents."},
    { LLDB_OPT_SET_3, false, "python-script", 's', required_argument, NULL, 0, eArgTypeName, "Give a one-liner Python script as part of the command."},
    { LLDB_OPT_SET_3, false, "python-function", 'F', required_argument, NULL, 0, eArgTypeName, "Give the name of a Python function to use for this type."},
    { LLDB_OPT_SET_3, false, "input-python", 'P', no_argument, NULL, 0, eArgTypeNone, "Input Python code to use for this type manually."},
    { LLDB_OPT_SET_2 | LLDB_OPT_SET_3,   false, "expand", 'e', no_argument, NULL, 0, eArgTypeNone,    "Expand aggregate data types to show children on separate lines."},
    { LLDB_OPT_SET_2 | LLDB_OPT_SET_3,   false, "name", 'n', required_argument, NULL, 0, eArgTypeName,    "A name for this summary string."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectTypeSummaryDelete
//-------------------------------------------------------------------------

class CommandObjectTypeSummaryDelete : public CommandObject
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
            
            switch (short_option)
            {
                case 'a':
                    m_delete_all = true;
                    break;
                case 'w':
                    m_category = ConstString(option_arg).GetCString();
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
            m_delete_all = false;
            m_category = NULL;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        bool m_delete_all;
        const char* m_category;
        
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    static bool
    PerCategoryCallback(void* param,
                        const char* cate_name,
                        const FormatCategory::SharedPointer& cate)
    {
        const char* name = (const char*)param;
        cate->Delete(name, FormatCategory::eSummary | FormatCategory::eRegexSummary);
        return true;
    }

public:
    CommandObjectTypeSummaryDelete (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type summary delete",
                   "Delete an existing summary style for a type.",
                   NULL), m_options(interpreter)
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
        
        if (!typeCS)
        {
            result.AppendError("empty typenames not allowed");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        if (m_options.m_delete_all)
        {
            Debugger::Formatting::Categories::LoopThrough(PerCategoryCallback, (void*)typeCS.GetCString());
            result.SetStatus(eReturnStatusSuccessFinishNoResult);
            return result.Succeeded();
        }
        
        lldb::FormatCategorySP category;
        Debugger::Formatting::Categories::Get(ConstString(m_options.m_category), category);
        
        bool delete_category = category->DeleteSummaries(typeCS.GetCString());
        bool delete_named = Debugger::Formatting::NamedSummaryFormats::Delete(typeCS);
        
        if (delete_category || delete_named)
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

OptionDefinition
CommandObjectTypeSummaryDelete::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "all", 'a', no_argument, NULL, 0, eArgTypeNone,  "Delete from every category."},
    { LLDB_OPT_SET_2, false, "category", 'w', required_argument, NULL, 0, eArgTypeName,  "Delete from given category."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

class CommandObjectTypeSummaryClear : public CommandObject
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
            
            switch (short_option)
            {
                case 'a':
                    m_delete_all = true;
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
            m_delete_all = false;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        bool m_delete_all;
        bool m_delete_named;
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    static bool
    PerCategoryCallback(void* param,
                        const char* cate_name,
                        const FormatCategory::SharedPointer& cate)
    {
        cate->Summary()->Clear();
        cate->RegexSummary()->Clear();
        return true;
        
    }
    
public:
    CommandObjectTypeSummaryClear (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type summary clear",
                   "Delete all existing summary styles.",
                   NULL), m_options(interpreter)
    {
    }
    
    ~CommandObjectTypeSummaryClear ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        
        if (m_options.m_delete_all)
            Debugger::Formatting::Categories::LoopThrough(PerCategoryCallback, NULL);
        
        else
        {        
            lldb::FormatCategorySP category;
            if (command.GetArgumentCount() > 0)
            {
                const char* cat_name = command.GetArgumentAtIndex(0);
                ConstString cat_nameCS(cat_name);
                Debugger::Formatting::Categories::Get(cat_nameCS, category);
            }
            else
                Debugger::Formatting::Categories::Get(ConstString(NULL), category);
            category->ClearSummaries();
        }
        
        Debugger::Formatting::NamedSummaryFormats::Clear();
        
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
};

OptionDefinition
CommandObjectTypeSummaryClear::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "all", 'a', no_argument, NULL, 0, eArgTypeNone,  "Clear every category."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectTypeSummaryList
//-------------------------------------------------------------------------

bool CommandObjectTypeSummaryList_LoopCallback(void* pt2self, const char* type, const StringSummaryFormat::SharedPointer& entry);
bool CommandObjectTypeRXSummaryList_LoopCallback(void* pt2self, lldb::RegularExpressionSP regex, const StringSummaryFormat::SharedPointer& entry);

class CommandObjectTypeSummaryList;

struct CommandObjectTypeSummaryList_LoopCallbackParam {
    CommandObjectTypeSummaryList* self;
    CommandReturnObject* result;
    RegularExpression* regex;
    RegularExpression* cate_regex;
    CommandObjectTypeSummaryList_LoopCallbackParam(CommandObjectTypeSummaryList* S, CommandReturnObject* R,
                                                  RegularExpression* X = NULL,
                                                  RegularExpression* CX = NULL) : self(S), result(R), regex(X), cate_regex(CX) {}
};

class CommandObjectTypeSummaryList : public CommandObject
{
    
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
            
            switch (short_option)
            {
                case 'w':
                    m_category_regex = std::string(option_arg);
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
            m_category_regex = "";
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        std::string m_category_regex;
        
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
public:
    CommandObjectTypeSummaryList (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type summary list",
                   "Show a list of current summary styles.",
                   NULL), m_options(interpreter)
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
        RegularExpression* cate_regex = 
        m_options.m_category_regex.empty() ? NULL :
        new RegularExpression(m_options.m_category_regex.c_str());
        
        if (argc == 1) {
            RegularExpression* regex = new RegularExpression(command.GetArgumentAtIndex(0));
            regex->Compile(command.GetArgumentAtIndex(0));
            param = new CommandObjectTypeSummaryList_LoopCallbackParam(this,&result,regex,cate_regex);
        }
        else
            param = new CommandObjectTypeSummaryList_LoopCallbackParam(this,&result,NULL,cate_regex);
        
        Debugger::Formatting::Categories::LoopThrough(PerCategoryCallback,param);
                
        if (Debugger::Formatting::NamedSummaryFormats::GetCount() > 0)
        {
            result.GetOutputStream().Printf("Named summaries:\n");
            if (argc == 1) {
                RegularExpression* regex = new RegularExpression(command.GetArgumentAtIndex(0));
                regex->Compile(command.GetArgumentAtIndex(0));
                param = new CommandObjectTypeSummaryList_LoopCallbackParam(this,&result,regex);
            }
            else
                param = new CommandObjectTypeSummaryList_LoopCallbackParam(this,&result);
            Debugger::Formatting::NamedSummaryFormats::LoopThrough(CommandObjectTypeSummaryList_LoopCallback, param);
            delete param;
        }
        
        if (cate_regex)
            delete cate_regex;
        
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
private:
    
    static bool
    PerCategoryCallback(void* param_vp,
                        const char* cate_name,
                        const FormatCategory::SharedPointer& cate)
    {
        
        CommandObjectTypeSummaryList_LoopCallbackParam* param = 
            (CommandObjectTypeSummaryList_LoopCallbackParam*)param_vp;
        CommandReturnObject* result = param->result;
        
        // if the category is disabled or empty and there is no regex, just skip it
        if ((cate->IsEnabled() == false || cate->GetCount() == 0) && param->cate_regex == NULL)
            return true;
        
        // if we have a regex and this category does not match it, just skip it
        if(param->cate_regex != NULL && param->cate_regex->Execute(cate_name) == false)
            return true;
        
        result->GetOutputStream().Printf("-----------------------\nCategory: %s (%s)\n-----------------------\n",
                                         cate_name,
                                         (cate->IsEnabled() ? "enabled" : "disabled"));
                
        cate->Summary()->LoopThrough(CommandObjectTypeSummaryList_LoopCallback, param_vp);
        
        if (cate->RegexSummary()->GetCount() > 0)
        {
            result->GetOutputStream().Printf("Regex-based summaries (slower):\n");
            cate->RegexSummary()->LoopThrough(CommandObjectTypeRXSummaryList_LoopCallback, param_vp);
        }
        return true;
    }
    
    bool
    LoopCallback (const char* type,
                  const SummaryFormat::SharedPointer& entry,
                  RegularExpression* regex,
                  CommandReturnObject *result)
    {
        if (regex == NULL || regex->Execute(type)) 
                result->GetOutputStream().Printf ("%s: %s\n", type, entry->GetDescription().c_str());
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
    CommandObjectTypeSummaryList_LoopCallbackParam* param = (CommandObjectTypeSummaryList_LoopCallbackParam*)pt2self;
    return param->self->LoopCallback(regex->GetText(), entry, param->regex, param->result);
}

OptionDefinition
CommandObjectTypeSummaryList::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "category-regex", 'w', required_argument, NULL, 0, eArgTypeName,  "Only show categories matching this filter."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectTypeCategoryEnable
//-------------------------------------------------------------------------

class CommandObjectTypeCategoryEnable : public CommandObject
{
public:
    CommandObjectTypeCategoryEnable (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type category enable",
                   "Enable a category as a source of summaries.",
                   NULL)
    {
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatPlus;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
        
    }
    
    ~CommandObjectTypeCategoryEnable ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        if (argc < 1)
        {
            result.AppendErrorWithFormat ("%s takes 1 or more args.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        for (int i = argc - 1; i >= 0; i--)
        {
            const char* typeA = command.GetArgumentAtIndex(i);
            ConstString typeCS(typeA);
            
            if (!typeCS)
            {
                result.AppendError("empty category name not allowed");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            Debugger::Formatting::Categories::Enable(typeCS);
        }
        
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
};

//-------------------------------------------------------------------------
// CommandObjectTypeCategoryDelete
//-------------------------------------------------------------------------

class CommandObjectTypeCategoryDelete : public CommandObject
{
public:
    CommandObjectTypeCategoryDelete (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type category delete",
                   "Delete a category and all associated summaries.",
                   NULL)
    {
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatPlus;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
        
    }
    
    ~CommandObjectTypeCategoryDelete ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        if (argc < 1)
        {
            result.AppendErrorWithFormat ("%s takes 1 or more arg.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        bool success = true;
        
        // the order is not relevant here
        for (int i = argc - 1; i >= 0; i--)
        {
            const char* typeA = command.GetArgumentAtIndex(i);
            ConstString typeCS(typeA);
            
            if (!typeCS)
            {
                result.AppendError("empty category name not allowed");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            if (!Debugger::Formatting::Categories::Delete(typeCS))
                success = false; // keep deleting even if we hit an error
        }
        if (success)
        {
            result.SetStatus(eReturnStatusSuccessFinishResult);
            return result.Succeeded();
        }
        else
        {
            result.AppendError("cannot delete one or more categories\n");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
    }
};

//-------------------------------------------------------------------------
// CommandObjectTypeCategoryDisable
//-------------------------------------------------------------------------

class CommandObjectTypeCategoryDisable : public CommandObject
{
public:
    CommandObjectTypeCategoryDisable (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type category disable",
                   "Disable a category as a source of summaries.",
                   NULL)
    {
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatPlus;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
        
    }
    
    ~CommandObjectTypeCategoryDisable ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        if (argc < 1)
        {
            result.AppendErrorWithFormat ("%s takes 1 or more args.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        // the order is not relevant here
        for (int i = argc - 1; i >= 0; i--)
        {
            const char* typeA = command.GetArgumentAtIndex(i);
            ConstString typeCS(typeA);
            
            if (!typeCS)
            {
                result.AppendError("empty category name not allowed");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            Debugger::Formatting::Categories::Disable(typeCS);
        }

        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
};

//-------------------------------------------------------------------------
// CommandObjectTypeCategoryList
//-------------------------------------------------------------------------

class CommandObjectTypeCategoryList : public CommandObject
{
private:
    
    struct CommandObjectTypeCategoryList_CallbackParam
    {
        CommandReturnObject* result;
        RegularExpression* regex;
        
        CommandObjectTypeCategoryList_CallbackParam(CommandReturnObject* res,
                                                    RegularExpression* rex = NULL) :
        result(res),
        regex(rex)
        {
        }
        
    };
    
    static bool
    PerCategoryCallback(void* param_vp,
                        const char* cate_name,
                        const FormatCategory::SharedPointer& cate)
    {
        CommandObjectTypeCategoryList_CallbackParam* param =
            (CommandObjectTypeCategoryList_CallbackParam*)param_vp;
        CommandReturnObject* result = param->result;
        RegularExpression* regex = param->regex;
        
        if (regex == NULL || regex->Execute(cate_name))
            result->GetOutputStream().Printf("Category %s is%s enabled\n",
                                       cate_name,
                                       (cate->IsEnabled() ? "" : " not"));
        return true;
    }
public:
    CommandObjectTypeCategoryList (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type category list",
                   "Provide a list of all existing categories.",
                   NULL)
    {
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatOptional;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
    }
    
    ~CommandObjectTypeCategoryList ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        RegularExpression* regex = NULL;
        
        if (argc == 0)
            ;
        else if (argc == 1)
            regex = new RegularExpression(command.GetArgumentAtIndex(0));
        else
        {
            result.AppendErrorWithFormat ("%s takes 0 or one arg.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        CommandObjectTypeCategoryList_CallbackParam param(&result,
                                                          regex);
        
        Debugger::Formatting::Categories::LoopThrough(PerCategoryCallback, &param);
        
        if (regex)
            delete regex;
        
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
};

//-------------------------------------------------------------------------
// CommandObjectTypeSynthList
//-------------------------------------------------------------------------

bool CommandObjectTypeSynthList_LoopCallback(void* pt2self, const char* type, const SyntheticFilter::SharedPointer& entry);
bool CommandObjectTypeSynthRXList_LoopCallback(void* pt2self, lldb::RegularExpressionSP regex, const SyntheticFilter::SharedPointer& entry);

class CommandObjectTypeSynthList;

struct CommandObjectTypeSynthList_LoopCallbackParam {
    CommandObjectTypeSynthList* self;
    CommandReturnObject* result;
    RegularExpression* regex;
    RegularExpression* cate_regex;
    CommandObjectTypeSynthList_LoopCallbackParam(CommandObjectTypeSynthList* S, CommandReturnObject* R,
                                                 RegularExpression* X = NULL,
                                                 RegularExpression* CX = NULL) : self(S), result(R), regex(X), cate_regex(CX) {}
};

class CommandObjectTypeSynthList : public CommandObject
{
    
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
            
            switch (short_option)
            {
                case 'w':
                    m_category_regex = std::string(option_arg);
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
            m_category_regex = "";
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        std::string m_category_regex;
        
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
public:
    CommandObjectTypeSynthList (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type synth list",
                   "Show a list of current synthetic providers.",
                   NULL), m_options(interpreter)
    {
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatOptional;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
    }
    
    ~CommandObjectTypeSynthList ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        CommandObjectTypeSynthList_LoopCallbackParam *param;
        RegularExpression* cate_regex = 
        m_options.m_category_regex.empty() ? NULL :
        new RegularExpression(m_options.m_category_regex.c_str());
        
        if (argc == 1) {
            RegularExpression* regex = new RegularExpression(command.GetArgumentAtIndex(0));
            regex->Compile(command.GetArgumentAtIndex(0));
            param = new CommandObjectTypeSynthList_LoopCallbackParam(this,&result,regex,cate_regex);
        }
        else
            param = new CommandObjectTypeSynthList_LoopCallbackParam(this,&result,NULL,cate_regex);
        
        Debugger::Formatting::Categories::LoopThrough(PerCategoryCallback,param);
                
        if (cate_regex)
            delete cate_regex;
        
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
private:
    
    static bool
    PerCategoryCallback(void* param_vp,
                        const char* cate_name,
                        const FormatCategory::SharedPointer& cate)
    {
        
        CommandObjectTypeSynthList_LoopCallbackParam* param = 
        (CommandObjectTypeSynthList_LoopCallbackParam*)param_vp;
        CommandReturnObject* result = param->result;
        
        // if the category is disabled or empty and there is no regex, just skip it
        if ((cate->IsEnabled() == false || cate->Filter()->GetCount() == 0) && param->cate_regex == NULL)
            return true;
        
        // if we have a regex and this category does not match it, just skip it
        if(param->cate_regex != NULL && param->cate_regex->Execute(cate_name) == false)
            return true;
        
        result->GetOutputStream().Printf("-----------------------\nCategory: %s (%s)\n-----------------------\n",
                                         cate_name,
                                         (cate->IsEnabled() ? "enabled" : "disabled"));
        
        cate->Filter()->LoopThrough(CommandObjectTypeSynthList_LoopCallback, param_vp);
        
        if (cate->RegexFilter()->GetCount() > 0)
        {
            result->GetOutputStream().Printf("Regex-based filters (slower):\n");
            cate->RegexFilter()->LoopThrough(CommandObjectTypeSynthRXList_LoopCallback, param_vp);
        }
        
        return true;
    }
    
    bool
    LoopCallback (const char* type,
                  const SyntheticFilter::SharedPointer& entry,
                  RegularExpression* regex,
                  CommandReturnObject *result)
    {
        if (regex == NULL || regex->Execute(type)) 
            result->GetOutputStream().Printf ("%s: %s\n", type, entry->GetDescription().c_str());
        return true;
    }
    
    friend bool CommandObjectTypeSynthList_LoopCallback(void* pt2self, const char* type, const SyntheticFilter::SharedPointer& entry);
    friend bool CommandObjectTypeSynthRXList_LoopCallback(void* pt2self, lldb::RegularExpressionSP regex, const SyntheticFilter::SharedPointer& entry);
};

bool
CommandObjectTypeSynthList_LoopCallback (void* pt2self,
                                         const char* type,
                                         const SyntheticFilter::SharedPointer& entry)
{
    CommandObjectTypeSynthList_LoopCallbackParam* param = (CommandObjectTypeSynthList_LoopCallbackParam*)pt2self;
    return param->self->LoopCallback(type, entry, param->regex, param->result);
}

bool
CommandObjectTypeSynthRXList_LoopCallback (void* pt2self,
                                         lldb::RegularExpressionSP regex,
                                         const SyntheticFilter::SharedPointer& entry)
{
    CommandObjectTypeSynthList_LoopCallbackParam* param = (CommandObjectTypeSynthList_LoopCallbackParam*)pt2self;
    return param->self->LoopCallback(regex->GetText(), entry, param->regex, param->result);
}


OptionDefinition
CommandObjectTypeSynthList::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "category-regex", 'w', required_argument, NULL, 0, eArgTypeName,  "Only show categories matching this filter."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectTypeSynthDelete
//-------------------------------------------------------------------------

class CommandObjectTypeSynthDelete : public CommandObject
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
            
            switch (short_option)
            {
                case 'a':
                    m_delete_all = true;
                    break;
                case 'w':
                    m_category = ConstString(option_arg).GetCString();
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
            m_delete_all = false;
            m_category = NULL;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        bool m_delete_all;
        const char* m_category;
        
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    static bool
    PerCategoryCallback(void* param,
                        const char* cate_name,
                        const FormatCategory::SharedPointer& cate)
    {
        const char* name = (const char*)param;
        return cate->Delete(name, FormatCategory::eFilter | FormatCategory::eRegexFilter);
    }
    
public:
    CommandObjectTypeSynthDelete (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type synth delete",
                   "Delete an existing synthetic provider for a type.",
                   NULL), m_options(interpreter)
    {
        CommandArgumentEntry type_arg;
        CommandArgumentData type_style_arg;
        
        type_style_arg.arg_type = eArgTypeName;
        type_style_arg.arg_repetition = eArgRepeatPlain;
        
        type_arg.push_back (type_style_arg);
        
        m_arguments.push_back (type_arg);
        
    }
    
    ~CommandObjectTypeSynthDelete ()
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
        
        if (!typeCS)
        {
            result.AppendError("empty typenames not allowed");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        if (m_options.m_delete_all)
        {
            Debugger::Formatting::Categories::LoopThrough(PerCategoryCallback, (void*)typeCS.GetCString());
            result.SetStatus(eReturnStatusSuccessFinishNoResult);
            return result.Succeeded();
        }
        
        lldb::FormatCategorySP category;
        Debugger::Formatting::Categories::Get(ConstString(m_options.m_category), category);
        
        bool delete_category = category->Filter()->Delete(typeCS.GetCString());
        delete_category = category->RegexFilter()->Delete(typeCS.GetCString()) || delete_category;
        
        if (delete_category)
        {
            result.SetStatus(eReturnStatusSuccessFinishNoResult);
            return result.Succeeded();
        }
        else
        {
            result.AppendErrorWithFormat ("no custom synthetic provider for %s.\n", typeA);
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
    }
};

OptionDefinition
CommandObjectTypeSynthDelete::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "all", 'a', no_argument, NULL, 0, eArgTypeNone,  "Delete from every category."},
    { LLDB_OPT_SET_2, false, "category", 'w', required_argument, NULL, 0, eArgTypeName,  "Delete from given category."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectTypeSynthClear
//-------------------------------------------------------------------------

class CommandObjectTypeSynthClear : public CommandObject
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
            
            switch (short_option)
            {
                case 'a':
                    m_delete_all = true;
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
            m_delete_all = false;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        bool m_delete_all;
        bool m_delete_named;
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    static bool
    PerCategoryCallback(void* param,
                        const char* cate_name,
                        const FormatCategory::SharedPointer& cate)
    {
        cate->Clear(FormatCategory::eFilter | FormatCategory::eRegexFilter);
        return true;
        
    }
    
public:
    CommandObjectTypeSynthClear (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "type synth clear",
                   "Delete all existing synthetic providers.",
                   NULL), m_options(interpreter)
    {
    }
    
    ~CommandObjectTypeSynthClear ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        
        if (m_options.m_delete_all)
            Debugger::Formatting::Categories::LoopThrough(PerCategoryCallback, NULL);
        
        else
        {        
            lldb::FormatCategorySP category;
            if (command.GetArgumentCount() > 0)
            {
                const char* cat_name = command.GetArgumentAtIndex(0);
                ConstString cat_nameCS(cat_name);
                Debugger::Formatting::Categories::Get(cat_nameCS, category);
            }
            else
                Debugger::Formatting::Categories::Get(ConstString(NULL), category);
            category->Filter()->Clear();
            category->RegexFilter()->Clear();
        }
        
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
    
};

OptionDefinition
CommandObjectTypeSynthClear::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "all", 'a', no_argument, NULL, 0, eArgTypeNone,  "Clear every category."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectTypeSynthAdd
//-------------------------------------------------------------------------

static const char *g_synth_addreader_instructions =   "Enter your Python command(s). Type 'DONE' to end.\n"
                                                      "You must define a Python class with these methods:\n"
                                                      "     def __init__(self, valobj, dict):\n"
                                                      "     def num_children(self):\n"
                                                      "     def get_child_at_index(self, index):\n"
                                                      "     def get_child_index(self, name):\n"
                                                      "Optionally, you can also define a method:\n"
                                                      "     def update(self):\n"
                                                      "if your synthetic provider is holding on to any per-object state variables (currently, this is not implemented because of the way LLDB handles instances of SBValue and you should not rely on object persistence and per-object state)\n"
                                                      "class synthProvider:";

class TypeSynthAddInputReader : public InputReaderEZ
{
private:
    DISALLOW_COPY_AND_ASSIGN (TypeSynthAddInputReader);
public:
    TypeSynthAddInputReader(Debugger& debugger) : 
    InputReaderEZ(debugger)
    {}
    
    virtual
    ~TypeSynthAddInputReader()
    {
    }
    
    virtual void ActivateHandler(HandlerData& data)
    {
        StreamSP out_stream = data.GetOutStream();
        bool batch_mode = data.GetBatchMode();
        if (!batch_mode)
        {
            out_stream->Printf ("%s\n", g_synth_addreader_instructions);
            if (data.reader.GetPrompt())
                out_stream->Printf ("%s", data.reader.GetPrompt());
            out_stream->Flush();
        }
    }
    
    virtual void ReactivateHandler(HandlerData& data)
    {
        StreamSP out_stream = data.GetOutStream();
        bool batch_mode = data.GetBatchMode();
        if (data.reader.GetPrompt() && !batch_mode)
        {
            out_stream->Printf ("%s", data.reader.GetPrompt());
            out_stream->Flush();
        }
    }
    virtual void GotTokenHandler(HandlerData& data)
    {
        StreamSP out_stream = data.GetOutStream();
        bool batch_mode = data.GetBatchMode();
        if (data.bytes && data.bytes_len && data.baton)
        {
            ((SynthAddOptions*)data.baton)->m_user_source.AppendString(data.bytes, data.bytes_len);
        }
        if (!data.reader.IsDone() && data.reader.GetPrompt() && !batch_mode)
        {
            out_stream->Printf ("%s", data.reader.GetPrompt());
            out_stream->Flush();
        }
    }
    virtual void InterruptHandler(HandlerData& data)
    {
        StreamSP out_stream = data.GetOutStream();
        bool batch_mode = data.GetBatchMode();
        data.reader.SetIsDone (true);
        if (!batch_mode)
        {
            out_stream->Printf ("Warning: No command attached to breakpoint.\n");
            out_stream->Flush();
        }
    }
    virtual void EOFHandler(HandlerData& data)
    {
        data.reader.SetIsDone (true);
    }
    virtual void DoneHandler(HandlerData& data)
    {
        StreamSP out_stream = data.GetOutStream();
        SynthAddOptions *options_ptr = ((SynthAddOptions*)data.baton);
        if (!options_ptr)
        {
            out_stream->Printf ("Internal error #1: no script attached.\n");
            out_stream->Flush();
            return;
        }
        
        SynthAddOptions::SharedPointer options(options_ptr); // this will ensure that we get rid of the pointer when going out of scope
        
        ScriptInterpreter *interpreter = data.reader.GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
        if (!interpreter)
        {
            out_stream->Printf ("Internal error #2: no script attached.\n");
            out_stream->Flush();
            return;
        }
        StringList class_name_sl;
        if (!interpreter->GenerateTypeSynthClass (options->m_user_source, 
                                                  class_name_sl))
        {
            out_stream->Printf ("Internal error #3: no script attached.\n");
            out_stream->Flush();
            return;
        }
        if (class_name_sl.GetSize() == 0)
        {
            out_stream->Printf ("Internal error #4: no script attached.\n");
            out_stream->Flush();
            return;
        }
        const char *class_name = class_name_sl.GetStringAtIndex(0);
        if (!class_name || !class_name[0])
        {
            out_stream->Printf ("Internal error #5: no script attached.\n");
            out_stream->Flush();
            return;
        }

        // everything should be fine now, let's add the synth provider class
        
        SyntheticChildrenSP synth_provider;
        synth_provider.reset(new SyntheticScriptProvider(options->m_cascade,
                                                         options->m_skip_pointers,
                                                         options->m_skip_references,
                                                         std::string(class_name)));
        
        
        lldb::FormatCategorySP category;
        Debugger::Formatting::Categories::Get(ConstString(options->m_category), category);
        
        for (size_t i = 0; i < options->m_target_types.GetSize(); i++) {
            const char *type_name = options->m_target_types.GetStringAtIndex(i);
            ConstString typeCS(type_name);
            if (typeCS)
                CommandObjectTypeSynthAdd::AddSynth(typeCS,
                                                    synth_provider,
                                                    options->m_regex ? CommandObjectTypeSynthAdd::eRegexSynth : CommandObjectTypeSynthAdd::eRegularSynth,
                                                    options->m_category,
                                                    NULL);
            else
            {
                out_stream->Printf ("Internal error #6: no script attached.\n");
                out_stream->Flush();
                return;
            }
        }
    }
};

void
CommandObjectTypeSynthAdd::CollectPythonScript (SynthAddOptions *options,
                                                CommandReturnObject &result)
{
    InputReaderSP reader_sp (new TypeSynthAddInputReader(m_interpreter.GetDebugger()));
    if (reader_sp && options)
    {
        
        InputReaderEZ::InitializationParameters ipr;
        
        Error err (reader_sp->Initialize (ipr.SetBaton(options).SetPrompt("     ")));
        if (err.Success())
        {
            m_interpreter.GetDebugger().PushInputReader (reader_sp);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError (err.AsCString());
            result.SetStatus (eReturnStatusFailed);
        }
    }
    else
    {
        result.AppendError("out of memory");
        result.SetStatus (eReturnStatusFailed);
    }
}
    
bool
CommandObjectTypeSynthAdd::Execute_HandwritePython (Args& command, CommandReturnObject &result)
{
    SynthAddOptions *options = new SynthAddOptions ( m_options.m_skip_pointers,
                                                     m_options.m_skip_references,
                                                     m_options.m_cascade,
                                                     m_options.m_regex,
                                                     m_options.m_category);
    
    const size_t argc = command.GetArgumentCount();
    
    for (size_t i = 0; i < argc; i++) {
        const char* typeA = command.GetArgumentAtIndex(i);
        if (typeA && *typeA)
            options->m_target_types << typeA;
        else
        {
            result.AppendError("empty typenames not allowed");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
    }
    
    CollectPythonScript(options,result);
    return result.Succeeded();
}
    
bool
CommandObjectTypeSynthAdd::Execute_ChildrenList (Args& command, CommandReturnObject &result)
{
    const size_t argc = command.GetArgumentCount();
    
    if (argc < 1)
    {
        result.AppendErrorWithFormat ("%s takes one or more args.\n", m_cmd_name.c_str());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
    
    if (m_options.m_expr_paths.size() == 0)
    {
        result.AppendErrorWithFormat ("%s needs one or more children.\n", m_cmd_name.c_str());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
    
    SyntheticChildrenSP entry;
    
    SyntheticFilter* impl = new SyntheticFilter(m_options.m_cascade,
                                                m_options.m_skip_pointers,
                                                m_options.m_skip_references);
    
    entry.reset(impl);
    
    // go through the expression paths
    CommandOptions::ExpressionPathsIterator begin, end = m_options.m_expr_paths.end();
    
    for (begin = m_options.m_expr_paths.begin(); begin != end; begin++)
        impl->AddExpressionPath(*begin);
    
    
    // now I have a valid provider, let's add it to every type
    
    lldb::FormatCategorySP category;
    Debugger::Formatting::Categories::Get(ConstString(m_options.m_category), category);
    
    for (size_t i = 0; i < argc; i++) {
        const char* typeA = command.GetArgumentAtIndex(i);
        ConstString typeCS(typeA);
        if (typeCS)
            AddSynth(typeCS,
                     entry,
                     m_options.m_regex ? eRegexSynth : eRegularSynth,
                     m_options.m_category,
                     NULL);
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
    
bool
CommandObjectTypeSynthAdd::Execute_PythonClass (Args& command, CommandReturnObject &result)
{
    const size_t argc = command.GetArgumentCount();
    
    if (argc < 1)
    {
        result.AppendErrorWithFormat ("%s takes one or more args.\n", m_cmd_name.c_str());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
    
    if (m_options.m_class_name.empty() && !m_options.m_input_python)
    {
        result.AppendErrorWithFormat ("%s needs either a Python class name or -P to directly input Python code.\n", m_cmd_name.c_str());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
    
    SyntheticChildrenSP entry;
    
    SyntheticScriptProvider* impl = new SyntheticScriptProvider(m_options.m_cascade,
                                                                m_options.m_skip_pointers,
                                                                m_options.m_skip_references,
                                                                m_options.m_class_name);
    
    entry.reset(impl);
    
    // now I have a valid provider, let's add it to every type
    
    lldb::FormatCategorySP category;
    Debugger::Formatting::Categories::Get(ConstString(m_options.m_category), category);
    
    for (size_t i = 0; i < argc; i++) {
        const char* typeA = command.GetArgumentAtIndex(i);
        ConstString typeCS(typeA);
        if (typeCS)
            AddSynth(typeCS,
                     entry,
                     m_options.m_regex ? eRegexSynth : eRegularSynth,
                     m_options.m_category,
                     NULL);
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
    
CommandObjectTypeSynthAdd::CommandObjectTypeSynthAdd (CommandInterpreter &interpreter) :
CommandObject (interpreter,
               "type synth add",
               "Add a new synthetic provider for a type.",
               NULL), m_options (interpreter)
{
    CommandArgumentEntry type_arg;
    CommandArgumentData type_style_arg;
    
    type_style_arg.arg_type = eArgTypeName;
    type_style_arg.arg_repetition = eArgRepeatPlus;
    
    type_arg.push_back (type_style_arg);
    
    m_arguments.push_back (type_arg);
    
}

bool
CommandObjectTypeSynthAdd::AddSynth(const ConstString& type_name,
         SyntheticChildrenSP entry,
         SynthFormatType type,
         const char* category_name,
         Error* error)
{
    lldb::FormatCategorySP category;
    Debugger::Formatting::Categories::Get(ConstString(category_name), category);
    
    if (type == eRegexSynth)
    {
        RegularExpressionSP typeRX(new RegularExpression());
        if (!typeRX->Compile(type_name.GetCString()))
        {
            if (error)
                error->SetErrorString("regex format error (maybe this is not really a regex?)");
            return false;
        }
        
        category->RegexFilter()->Delete(type_name.GetCString());
        category->RegexFilter()->Add(typeRX, entry);
        
        return true;
    }
    else
    {
        category->Filter()->Add(type_name.GetCString(), entry);
        return true;
    }
}
    
bool
CommandObjectTypeSynthAdd::Execute (Args& command, CommandReturnObject &result)
{
    if (m_options.handwrite_python)
        return Execute_HandwritePython(command, result);
    else if (m_options.is_class_based)
        return Execute_PythonClass(command, result);
    else if (m_options.has_child_list)
        return Execute_ChildrenList(command, result);
    else
    {
        result.AppendError("must either provide a children list, a Python class name, or use -P and type a Python class line-by-line");
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
}

OptionDefinition
CommandObjectTypeSynthAdd::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "cascade", 'C', required_argument, NULL, 0, eArgTypeBoolean,    "If true, cascade to derived typedefs."},
    { LLDB_OPT_SET_ALL, false, "skip-pointers", 'p', no_argument, NULL, 0, eArgTypeNone,         "Don't use this format for pointers-to-type objects."},
    { LLDB_OPT_SET_ALL, false, "skip-references", 'r', no_argument, NULL, 0, eArgTypeNone,         "Don't use this format for references-to-type objects."},
    { LLDB_OPT_SET_ALL, false, "category", 'w', required_argument, NULL, 0, eArgTypeName,         "Add this to the given category instead of the default one."},
    { LLDB_OPT_SET_1, false, "child", 'c', required_argument, NULL, 0, eArgTypeName,    "Include this expression path in the synthetic view."},
    { LLDB_OPT_SET_2, false, "python-class", 'l', required_argument, NULL, 0, eArgTypeName,    "Use this Python class to produce synthetic children."},
    { LLDB_OPT_SET_3, false, "input-python", 'P', no_argument, NULL, 0, eArgTypeNone,    "Type Python code to generate a class that provides synthetic children."},
    { LLDB_OPT_SET_ALL, false,  "regex", 'x', no_argument, NULL, 0, eArgTypeNone,    "Type names are actually regular expressions."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

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

class CommandObjectTypeSynth : public CommandObjectMultiword
{
public:
    CommandObjectTypeSynth (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "type synth",
                            "A set of commands for operating on synthetic type representations",
                            "type synth [<sub-command-options>] ")
    {
        LoadSubCommand ("add",           CommandObjectSP (new CommandObjectTypeSynthAdd (interpreter)));
        LoadSubCommand ("clear",         CommandObjectSP (new CommandObjectTypeSynthClear (interpreter)));
        LoadSubCommand ("delete",        CommandObjectSP (new CommandObjectTypeSynthDelete (interpreter)));
        LoadSubCommand ("list",          CommandObjectSP (new CommandObjectTypeSynthList (interpreter)));
    }
    
    
    ~CommandObjectTypeSynth ()
    {
    }
};

class CommandObjectTypeCategory : public CommandObjectMultiword
{
public:
    CommandObjectTypeCategory (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "type category",
                            "A set of commands for operating on categories",
                            "type category [<sub-command-options>] ")
    {
        LoadSubCommand ("enable",        CommandObjectSP (new CommandObjectTypeCategoryEnable (interpreter)));
        LoadSubCommand ("disable",       CommandObjectSP (new CommandObjectTypeCategoryDisable (interpreter)));
        LoadSubCommand ("delete",        CommandObjectSP (new CommandObjectTypeCategoryDelete (interpreter)));
        LoadSubCommand ("list",          CommandObjectSP (new CommandObjectTypeCategoryList (interpreter)));
    }
    
    
    ~CommandObjectTypeCategory ()
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
        LoadSubCommand ("add",           CommandObjectSP (new CommandObjectTypeSummaryAdd (interpreter)));
        LoadSubCommand ("clear",         CommandObjectSP (new CommandObjectTypeSummaryClear (interpreter)));
        LoadSubCommand ("delete",        CommandObjectSP (new CommandObjectTypeSummaryDelete (interpreter)));
        LoadSubCommand ("list",          CommandObjectSP (new CommandObjectTypeSummaryList (interpreter)));
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
    LoadSubCommand ("category",  CommandObjectSP (new CommandObjectTypeCategory (interpreter)));
    LoadSubCommand ("format",    CommandObjectSP (new CommandObjectTypeFormat (interpreter)));
    LoadSubCommand ("summary",   CommandObjectSP (new CommandObjectTypeSummary (interpreter)));
    LoadSubCommand ("synth",     CommandObjectSP (new CommandObjectTypeSynth (interpreter)));
}


CommandObjectType::~CommandObjectType ()
{
}


