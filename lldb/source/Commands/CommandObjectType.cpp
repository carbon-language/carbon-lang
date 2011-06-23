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
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectTypeAdd
//-------------------------------------------------------------------------

class CommandObjectTypeAdd : public CommandObject
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
    CommandObjectTypeAdd (CommandInterpreter &interpreter) :
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
    
    ~CommandObjectTypeAdd ()
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
        
        const char* formatA = command.GetArgumentAtIndex(0);
        ConstString formatCS(formatA);
        const char* formatU = formatCS.GetCString();
        lldb::Format format;
        uint32_t byte_size_ptr;
        Error fmt_error = Args::StringToFormat(formatU, format, &byte_size_ptr);
        
        if(fmt_error.Fail()) {
            result.AppendError(fmt_error.AsCString());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        // now I have a valid format, let's add it to every type
        
        for(int i = 1; i < argc; i++) {
            const char* typeA = command.GetArgumentAtIndex(i);
            ConstString typeCS(typeA);
            Debugger::AddFormatForType(typeCS, format, m_options.m_cascade);
        }
        
        
        return result.Succeeded();
    }
        
};

OptionDefinition
CommandObjectTypeAdd::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "cascade", 'c', required_argument, NULL, 0, eArgTypeBoolean,    "If true, cascade to derived typedefs."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectTypeDelete
//-------------------------------------------------------------------------

class CommandObjectTypeDelete : public CommandObject
{
public:
    CommandObjectTypeDelete (CommandInterpreter &interpreter) :
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
    
    ~CommandObjectTypeDelete ()
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
        
        if(Debugger::DeleteFormatForType(typeCS))
            return result.Succeeded();
        else
        {
            result.AppendErrorWithFormat ("no custom format for %s.\n", typeA);
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

    }
    
};

//-------------------------------------------------------------------------
// CommandObjectTypeList
//-------------------------------------------------------------------------

bool CommandObjectTypeList_LoopCallback(void* pt2self, const char* type, lldb::Format format, bool cascade);

class CommandObjectTypeList;

struct CommandObjectTypeList_LoopCallbackParam {
    CommandObjectTypeList* self;
    CommandReturnObject* result;
    RegularExpression* regex;
    CommandObjectTypeList_LoopCallbackParam(CommandObjectTypeList* S, CommandReturnObject* R,
                                            RegularExpression* X = NULL) : self(S), result(R), regex(X) {}
};

class CommandObjectTypeList : public CommandObject
{
public:
    CommandObjectTypeList (CommandInterpreter &interpreter) :
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
    
    ~CommandObjectTypeList ()
    {
    }
    
    bool
    Execute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        
        CommandObjectTypeList_LoopCallbackParam *param;
        
        if (argc == 1) {
            RegularExpression* regex = new RegularExpression(command.GetArgumentAtIndex(0));
            regex->Compile(command.GetArgumentAtIndex(0));
            param = new CommandObjectTypeList_LoopCallbackParam(this,&result,regex);
        }
        else
            param = new CommandObjectTypeList_LoopCallbackParam(this,&result);
        Debugger::LoopThroughFormatList(CommandObjectTypeList_LoopCallback, param);
        delete param;
        return result.Succeeded();
    }
    
private:
    
    bool
    LoopCallback (
                  const char* type,
                  lldb::Format format,
                  bool cascade,
                  RegularExpression* regex,
                  CommandReturnObject *result
                  )
    {
        if(regex && !regex->Execute(type)) return true;
        Stream &ostrm = result->GetOutputStream();
        ostrm.Printf("(%s) %scascading ",type, cascade ? "" : "not ");
        switch(format) {
            case eFormatBytes:
                ostrm.Printf("y\n");
                break;
            case eFormatBytesWithASCII:
                ostrm.Printf("Y\n");
                break;
            case eFormatBinary:
                ostrm.Printf("b\n");
                break;
            case eFormatBoolean:
                ostrm.Printf("B\n");
                break;
            case eFormatCharArray:
                ostrm.Printf("a\n");
                break;
            case eFormatChar:
                ostrm.Printf("c\n");
                break;
            case eFormatCharPrintable:
                ostrm.Printf("C\n");
                break;
            case eFormatOctal:
                ostrm.Printf("o\n");
                break;
            case eFormatOSType:
                ostrm.Printf("O\n");
                break;
            case eFormatDecimal:
                ostrm.Printf("i or d\n");
                break;
            case eFormatComplexInteger:
                ostrm.Printf("I\n");
                break;
            case eFormatUnsigned:
                ostrm.Printf("u\n");
                break;
            case eFormatHex:
                ostrm.Printf("x\n");
                break;
            case eFormatComplex:
                ostrm.Printf("X\n");
                break;
            case eFormatFloat:
                ostrm.Printf("f e or g\n");
                break;
            case eFormatPointer:
                ostrm.Printf("p\n");
                break;
            case eFormatCString:
                ostrm.Printf("s\n");
                break;
            default:
                ostrm.Printf("other\n");
                break;
        }
        return true;
    }
    
    friend bool CommandObjectTypeList_LoopCallback(void* pt2self, const char* type, lldb::Format format, bool cascade);
    
};

bool
CommandObjectTypeList_LoopCallback (
                                    void* pt2self,
                                    const char* type,
                                    lldb::Format format,
                                    bool cascade)
{
    CommandObjectTypeList_LoopCallbackParam* param = (CommandObjectTypeList_LoopCallbackParam*)pt2self;
    return param->self->LoopCallback(type, format, cascade, param->regex, param->result);
}

class CommandObjectTypeFormat : public CommandObjectMultiword
{
public:
    CommandObjectTypeFormat (CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter,
                                "type format",
                                "A set of commands for editing variable display options",
                                "type format [<sub-command-options>] ")
    {
        LoadSubCommand ("add",    CommandObjectSP (new CommandObjectTypeAdd (interpreter)));
        LoadSubCommand ("delete", CommandObjectSP (new CommandObjectTypeDelete (interpreter)));
        LoadSubCommand ("list",   CommandObjectSP (new CommandObjectTypeList (interpreter)));
    }


    ~CommandObjectTypeFormat ()
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
}


CommandObjectType::~CommandObjectType ()
{
}


