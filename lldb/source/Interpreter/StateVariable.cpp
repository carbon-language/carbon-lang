//===-- StateVariable.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandInterpreter.h"


#include "lldb/Interpreter/StateVariable.h"

using namespace lldb;
using namespace lldb_private;

// Variables with integer values.

StateVariable::StateVariable
(
    const char *name,
    int value,
    const char *help,
    Callback func_ptr
) :
    m_name (name),
    m_type (eTypeInteger),
    m_help_text (help),
    m_verification_func_ptr (func_ptr)
{
    m_int_value = value;
}

// Variables with boolean values.

StateVariable::StateVariable
(
    const char *name,
    bool value,
    const char *help,
    Callback func_ptr
 ) :
    m_name (name),
    m_type (eTypeBoolean),
    m_help_text (help),
    m_verification_func_ptr (func_ptr)
{
    m_int_value = value;
}

// Variables with string values.

StateVariable::StateVariable
(
    const char *name,
    const char *value,
    bool can_append,
    const char *help,
    Callback func_ptr
 ) :
    m_name (name),
    m_type (eTypeString),
    m_int_value (0),
    m_string_values (),
    m_help_text (help),
    m_verification_func_ptr (func_ptr)
{
    m_string_values.AppendArgument(value);
}

// Variables with array of strings values.

StateVariable::StateVariable
(
    const char *name,
    const Args *args,
    const char *help,
    Callback func_ptr
 ) :
    m_name (name),
    m_type (eTypeStringArray),
    m_help_text (help),
    m_string_values(),
    m_verification_func_ptr (func_ptr)
{
    if (args)
        m_string_values = *args;
}

StateVariable::~StateVariable ()
{
}

const char *
StateVariable::GetName () const
{
    return m_name.c_str();
}

StateVariable::Type
StateVariable::GetType () const
{
    return m_type;
}

int
StateVariable::GetIntValue () const
{
    return m_int_value;
}

bool
StateVariable::GetBoolValue () const
{
    return m_int_value;
}

const char *
StateVariable::GetStringValue () const
{
    return m_string_values.GetArgumentAtIndex(0);
}

const Args &
StateVariable::GetArgs () const
{
    return m_string_values;
}

Args &
StateVariable::GetArgs ()
{
    return m_string_values;
}

const char *
StateVariable::GetHelp () const
{
    return m_help_text.c_str();
}

void
StateVariable::SetHelp (const char *help)
{
    m_help_text = help;
}

void
StateVariable::AppendVariableInformation (CommandReturnObject &result)
{
    switch (m_type)
    {
    case eTypeBoolean:
        if (m_int_value)
            result.AppendMessageWithFormat ("    %s (bool) = True\n", m_name.c_str());
        else
            result.AppendMessageWithFormat ("    %s (bool) = False\n", m_name.c_str());
        break;

    case eTypeInteger:
        result.AppendMessageWithFormat ("    %s (int)  = %d\n", m_name.c_str(), m_int_value);
        break;

    case eTypeString:
        {
            const char *cstr = m_string_values.GetArgumentAtIndex(0);
            if (cstr && cstr[0])
                result.AppendMessageWithFormat ("    %s (str)  = '%s'\n", m_name.c_str(), cstr);
            else
                result.AppendMessageWithFormat ("    %s (str)  = <no value>\n", m_name.c_str());
        }
        break;

    case eTypeStringArray:
        {
            const size_t argc = m_string_values.GetArgumentCount();
            result.AppendMessageWithFormat ("    %s (string vector):\n", m_name.c_str());
            for (size_t i = 0; i < argc; ++i)
                result.AppendMessageWithFormat ("      [%d] %s\n", i, m_string_values.GetArgumentAtIndex(i));
        }
        break;

    default:
        break;
    }
}

void
StateVariable::SetStringValue (const char *new_value)
{
    if (m_string_values.GetArgumentCount() > 0)
        m_string_values.ReplaceArgumentAtIndex(0, new_value);
    else
        m_string_values.AppendArgument(new_value);
}

void
StateVariable::SetIntValue (int new_value)
{
    m_int_value = new_value;
}

void
StateVariable::SetBoolValue (bool new_value)
{
    m_int_value = new_value;
}

void
StateVariable::AppendStringValue (const char *cstr)
{
    if (cstr && cstr[0])
    {
        if (m_string_values.GetArgumentCount() == 0)
        {
            m_string_values.AppendArgument(cstr);
        }
        else
        {
            const char *curr_arg = m_string_values.GetArgumentAtIndex(0);
            if (curr_arg != NULL)
            {
                std::string new_arg_str(curr_arg);
                new_arg_str += " ";
                new_arg_str += cstr;
                m_string_values.ReplaceArgumentAtIndex(0, new_arg_str.c_str());
            }
            else
            {
                m_string_values.ReplaceArgumentAtIndex(0, cstr);
            }
        }
    }
}

bool
StateVariable::VerifyValue (CommandInterpreter *interpreter, void *data, CommandReturnObject &result)
{
    return (*m_verification_func_ptr) (interpreter, data, result);
}

//void
//StateVariable::SetArrayValue (STLStringArray &new_value)
//{
//    m_string_values.AppendArgument.append(cstr);
//
//    if (m_array_value != NULL)
//    {
//      if (m_array_value->size() > 0)
//      {
//          m_array_value->clear();
//      }
//    }
//    else
//        m_array_value = new STLStringArray;
//
//    for (int i = 0; i < new_value.size(); ++i)
//        m_array_value->push_back (new_value[i]);
//}
//

void
StateVariable::ArrayClearValues ()
{
    m_string_values.Clear();
}


void
StateVariable::ArrayAppendValue (const char *cstr)
{
    m_string_values.AppendArgument(cstr);
}


bool
StateVariable::HasVerifyFunction ()
{
    return (m_verification_func_ptr != NULL);
}

// Verification functions for various command interpreter variables.

bool
StateVariable::VerifyScriptLanguage (CommandInterpreter *interpreter, void *data, CommandReturnObject &result)
{
    bool valid_lang = true;
    interpreter->SetScriptLanguage (Args::StringToScriptLanguage((char *) data, eScriptLanguageDefault, &valid_lang));
    return valid_lang;
}

bool
StateVariable::BroadcastPromptChange (CommandInterpreter *interpreter, void *data, CommandReturnObject &result)
{
    char *prompt = (char *) data;
    if (prompt != NULL)
    {
        std::string tmp_prompt = prompt ;
        int len = tmp_prompt.size();
        if (len > 1
            && (tmp_prompt[0] == '\'' || tmp_prompt[0] == '"')
            && (tmp_prompt[len-1] == tmp_prompt[0]))
        {
            tmp_prompt = tmp_prompt.substr(1,len-2);
        }
        len = tmp_prompt.size();
        if (tmp_prompt[len-1] != ' ')
            tmp_prompt.append(" ");
        strcpy (prompt, tmp_prompt.c_str());
        data = (void *) prompt;
    }
    EventSP new_event_sp;
    new_event_sp.reset (new Event(CommandInterpreter::eBroadcastBitResetPrompt, new EventDataBytes (prompt)));
    interpreter->BroadcastEvent (new_event_sp);

    return true;
}

