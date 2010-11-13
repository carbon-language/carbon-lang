//===-- CommandObjectCall.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectCall.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangExpressionVariable.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/StackFrame.h"

using namespace lldb;
using namespace lldb_private;

// This command is a toy.  I'm just using it to have a way to construct the arguments to
// calling functions.
//

CommandObjectCall::CommandOptions::CommandOptions () :
    Options()
{
    // Keep only one place to reset the values to their defaults
    ResetOptionValues();
}


CommandObjectCall::CommandOptions::~CommandOptions ()
{
}

Error
CommandObjectCall::CommandOptions::SetOptionValue (int option_idx, const char *option_arg)
{
    Error error;

    char short_option = (char) m_getopt_table[option_idx].val;

    switch (short_option)
    {
    case 'l':
        if (language.SetLanguageFromCString (option_arg) == false)
        {
            error.SetErrorStringWithFormat("Invalid language option argument '%s'.\n", option_arg);
        }
        break;

    case 'g':
        debug = true;
        break;

    case 'f':
        error = Args::StringToFormat(option_arg,format);
        break;

    case 'n':
        noexecute = true;
        break;
            
    case 'a':
        use_abi = true;
        break;
            
    default:
        error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
        break;
    }

    return error;
}

void
CommandObjectCall::CommandOptions::ResetOptionValues ()
{
    Options::ResetOptionValues();
    language.Clear();
    debug = false;
    format = eFormatDefault;
    show_types = true;
    show_summary = true;
    noexecute = false;
    use_abi = false;
}

const lldb::OptionDefinition*
CommandObjectCall::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}

CommandObjectCall::CommandObjectCall () :
    CommandObject (
            "call",
            "Call a function.",
            //"call <return_type> <function-name> [[<arg1-type> <arg1-value>] ... <argn-type> <argn-value>] [<cmd-options>]",
            NULL,
            eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
{
    CommandArgumentEntry arg1;
    CommandArgumentEntry arg2;
    CommandArgumentEntry arg3;
    CommandArgumentData return_type_arg;
    CommandArgumentData function_name_arg;
    CommandArgumentData arg_type_arg;
    CommandArgumentData arg_value_arg;

    // Define the first (and only) variant of this arg.
    return_type_arg.arg_type = eArgTypeType;
    return_type_arg.arg_repetition = eArgRepeatPlain;

    arg1.push_back (return_type_arg);

    function_name_arg.arg_type = eArgTypeFunctionName;
    function_name_arg.arg_repetition = eArgTypePlain;

    arg2.push_back (function_name_arg);

    arg_type_arg.arg_type = eArgTypeArgType;
    arg_type_arg.arg_repetition = eArgRepeatPairRangeOptional;

    arg_value_arg.arg_type = eArgTypeValue;
    arg_value_arg.arg_repetition = eArgRepeatPairRangeOptional;

    arg3.push_back (arg_type_arg);
    arg3.push_back (arg_value_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back (arg1);
    m_arguments.push_back (arg2);
    m_arguments.push_back (arg3);
}

CommandObjectCall::~CommandObjectCall ()
{
}

Options *
CommandObjectCall::GetOptions ()
{
    return &m_options;
}

bool
CommandObjectCall::Execute
(
    Args &command,
    CommandReturnObject &result
)
{
    ConstString target_triple;
    int num_args = command.GetArgumentCount();

    ExecutionContext exe_ctx(interpreter.GetDebugger().GetExecutionContext());
    if (exe_ctx.target)
        exe_ctx.target->GetTargetTriple(target_triple);

    if (!target_triple)
        target_triple = Host::GetTargetTriple ();

    if (exe_ctx.thread == NULL || exe_ctx.frame == NULL)
    {
        result.AppendError ("No currently selected thread and frame.");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    if (num_args < 2)
    {
        result.AppendErrorWithFormat ("Invalid usage, should be: %s.\n", GetSyntax());
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    if ((num_args - 2) %2 != 0)
    {
        result.AppendErrorWithFormat ("Invalid usage - unmatched args & types, should be: %s.\n", GetSyntax());
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    if (target_triple)
    {
        //const char *return_type = command.GetArgumentAtIndex(0);
        const char *function_name = command.GetArgumentAtIndex(1);
        // Look up the called function:
        
        Function *target_fn = NULL;
        
        SymbolContextList sc_list;
        
        exe_ctx.frame->GetSymbolContext(eSymbolContextEverything).FindFunctionsByName(ConstString(function_name), false, sc_list);

        if (sc_list.GetSize() > 0)
        {
            SymbolContext sc;
            sc_list.GetContextAtIndex(0, sc);
            target_fn = sc.function;
        }
        
        // FIXME: If target_fn is NULL, we should look up the name as a symbol and use it and the provided
        // return type.

        if (target_fn == NULL)
        {
            result.AppendErrorWithFormat ("Could not find function '%s'.\n", function_name);
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        ValueList value_list;
        // Okay, now parse arguments.  For now we only accept basic types.
        for (int i = 2; i < num_args; i+= 2)
        {
            const char *type_str = command.GetArgumentAtIndex(i);
            const char *value_str = command.GetArgumentAtIndex(i + 1);
            bool success;
            if (strcmp(type_str, "int") == 0
                || strcmp(type_str, "int32_t") == 0)
            {
                value_list.PushValue(Value(Args::StringToSInt32(value_str, 0, 0, &success)));
            }
            else if (strcmp (type_str, "int64_t") == 0)
            {
                value_list.PushValue(Value(Args::StringToSInt64(value_str, 0, 0, &success)));
            }
            else if (strcmp(type_str, "uint") == 0
                || strcmp(type_str, "uint32_t") == 0)
            {
                value_list.PushValue(Value(Args::StringToUInt32(value_str, 0, 0, &success)));
            }
            else if (strcmp (type_str, "uint64_t") == 0)
            {
                value_list.PushValue(Value(Args::StringToUInt64(value_str, 0, 0, &success)));
            }
            else if (strcmp (type_str, "cstr") == 0)
            {
                Value val ((intptr_t)value_str);
                val.SetValueType (Value::eValueTypeHostAddress);
                
                
                void *cstr_type = exe_ctx.target->GetScratchClangASTContext()->GetCStringType(true);
                val.SetContext (Value::eContextTypeClangType, cstr_type);
                value_list.PushValue(val);
                
                success = true;
            }

            if (!success)
            {
                result.AppendErrorWithFormat ("Could not convert value: '%s' to type '%s'.\n", value_str, type_str);
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        // Okay, we have the function and the argument list and the return type.  Now make a ClangFunction object and
        // run it:

        StreamString errors;
        ClangFunction clang_fun (target_triple.GetCString(), *target_fn, exe_ctx.target->GetScratchClangASTContext(), value_list);
        if (m_options.noexecute)
        {
            // Now write down the argument values for this call.
            lldb::addr_t args_addr = LLDB_INVALID_ADDRESS;
            if (!clang_fun.InsertFunction (exe_ctx, args_addr, errors))
            {
                result.AppendErrorWithFormat("Error inserting function: '%s'.\n", errors.GetData());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            else
            {
                result.Succeeded();
                return true;
            }
        }
        
        ClangFunction::ExecutionResults return_status;
        Value return_value;
        
        bool stop_others = true;
        return_status = clang_fun.ExecuteFunction(exe_ctx, errors, stop_others, NULL, return_value);

        // Now figure out what to do with the return value.
        if (return_status == ClangFunction::eExecutionSetupError)
        {
            result.AppendErrorWithFormat("Error setting up function execution: '%s'.\n", errors.GetData());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else if (return_status != ClangFunction::eExecutionCompleted)
        {
            result.AppendWarningWithFormat("Interrupted while calling function: '%s'.\n", errors.GetData());
            result.SetStatus(eReturnStatusSuccessFinishNoResult);
            return true;
        }
        else
        {
            // Now print out the result.
            result.GetOutputStream().Printf("Return value: ");
            return_value.Dump(&(result.GetOutputStream()));
            result.Succeeded();
        }

    }
    else
    {
        result.AppendError ("invalid target triple");
        result.SetStatus (eReturnStatusFailed);
    }
    return result.Succeeded();
}

lldb::OptionDefinition
CommandObjectCall::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "language",   'l', required_argument, NULL, 0, "[c|c++|objc|objc++]",          "Sets the language to use when parsing the expression."},
{ LLDB_OPT_SET_1, false, "format",     'f', required_argument, NULL, 0, "[ [bool|b] | [bin] | [char|c] | [oct|o] | [dec|i|d|u] | [hex|x] | [float|f] | [cstr|s] ]",  "Specify the format that the expression output should use."},
{ LLDB_OPT_SET_1, false, "debug",      'g', no_argument,       NULL, 0, NULL,                           "Enable verbose debug logging of the expression parsing and evaluation."},
{ LLDB_OPT_SET_1, false, "noexecute",  'n', no_argument,       NULL, 0, "no execute",                   "Only JIT and copy the wrapper & arguments, but don't execute."},
{ LLDB_OPT_SET_1, false, "use-abi",    'a', no_argument,       NULL, 0, NULL,                           "Use the ABI instead of the JIT to marshall arguments."},
{ 0, false, NULL, 0, 0, NULL, NULL, NULL, NULL }
};

