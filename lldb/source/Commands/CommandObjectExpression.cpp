//===-- CommandObjectExpression.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectExpression.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Expression/ClangExpressionVariable.h"
#include "lldb/Expression/ClangUserExpression.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Host/Host.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

CommandObjectExpression::CommandOptions::CommandOptions () :
    Options()
{
    // Keep only one place to reset the values to their defaults
    ResetOptionValues();
}


CommandObjectExpression::CommandOptions::~CommandOptions ()
{
}

Error
CommandObjectExpression::CommandOptions::SetOptionValue (int option_idx, const char *option_arg)
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
        error = Args::StringToFormat(option_arg, format);
        break;

    default:
        error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
        break;
    }

    return error;
}

void
CommandObjectExpression::CommandOptions::ResetOptionValues ()
{
    Options::ResetOptionValues();
    language.Clear();
    debug = false;
    format = eFormatDefault;
    show_types = true;
    show_summary = true;
}

const lldb::OptionDefinition*
CommandObjectExpression::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}

CommandObjectExpression::CommandObjectExpression () :
    CommandObject (
            "expression",
            "Evaluate a C expression in the current program context, using variables currently in scope.",
            "expression [<cmd-options>] <expr>"),
    m_expr_line_count (0),
    m_expr_lines ()
{
  SetHelpLong(
"Examples: \n\
\n\
   expr my_struct->a = my_array[3] \n\
   expr -f bin -- (index * 8) + 5 \n\
   expr char c[] = \"foo\"; c[0]\n");
}

CommandObjectExpression::~CommandObjectExpression ()
{
}

Options *
CommandObjectExpression::GetOptions ()
{
    return &m_options;
}


bool
CommandObjectExpression::Execute
(
    CommandInterpreter &interpreter,
    Args& command,
    CommandReturnObject &result
)
{
    return false;
}


size_t
CommandObjectExpression::MultiLineExpressionCallback
(
    void *baton, 
    InputReader &reader, 
    lldb::InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    CommandObjectExpression *cmd_object_expr = (CommandObjectExpression *) baton;

    switch (notification)
    {
    case eInputReaderActivate:
        reader.GetDebugger().GetOutputStream().Printf("%s\n", "Enter expressions, then terminate with an empty line to evaluate:");
        // Fall through
    case eInputReaderReactivate:
        //if (out_fh)
        //    reader.GetDebugger().GetOutputStream().Printf ("%3u: ", cmd_object_expr->m_expr_line_count);
        break;

    case eInputReaderDeactivate:
        break;

    case eInputReaderGotToken:
        ++cmd_object_expr->m_expr_line_count;
        if (bytes && bytes_len)
        {
            cmd_object_expr->m_expr_lines.append (bytes, bytes_len + 1);
        }

        if (bytes_len == 0)
            reader.SetIsDone(true);
        //else if (out_fh && !reader->IsDone())
        //    ::fprintf (out_fh, "%3u: ", cmd_object_expr->m_expr_line_count);
        break;
        
    case eInputReaderDone:
        {
            bool bare = false;
            cmd_object_expr->EvaluateExpression (cmd_object_expr->m_expr_lines.c_str(), 
                                                 bare, 
                                                 reader.GetDebugger().GetOutputStream(), 
                                                 reader.GetDebugger().GetErrorStream());
        }
        break;
    }

    return bytes_len;
}

bool
CommandObjectExpression::EvaluateExpression (const char *expr, bool bare, Stream &output_stream, Stream &error_stream,
                                             CommandReturnObject *result)
{
    ClangUserExpression user_expression (expr);
    
    if (!user_expression.Parse (error_stream, m_exe_ctx))
    {
        error_stream.Printf ("Couldn't parse the expresssion");
        return false;
    }
    
    ClangExpressionVariable *expr_result;
    
    if (!user_expression.Execute (error_stream, m_exe_ctx, expr_result))
    {
        error_stream.Printf ("Couldn't execute the expresssion");
        return false;
    }
        
    if (expr_result)
    {
        StreamString ss;
        
        Error rc = expr_result->Print (ss, 
                                       m_exe_ctx, 
                                       m_options.format,
                                       m_options.show_types,
                                       m_options.show_summary,
                                       m_options.debug);
        
        if (rc.Fail()) {
            error_stream.Printf ("Couldn't print result : %s\n", rc.AsCString());
            return false;
        }

        output_stream.PutCString(ss.GetString().c_str());
        if (result)
            result->SetStatus (eReturnStatusSuccessFinishResult);
    }
    else
    {
        error_stream.Printf ("Expression produced no result\n");
        if (result)
            result->SetStatus (eReturnStatusSuccessFinishNoResult);
    }
        
    return true;
}

bool
CommandObjectExpression::ExecuteRawCommandString
(
    CommandInterpreter &interpreter,
    const char *command,
    CommandReturnObject &result
)
{
    m_exe_ctx = interpreter.GetDebugger().GetExecutionContext();

    m_options.ResetOptionValues();

    const char * expr = NULL;

    if (command[0] == '\0')
    {
        m_expr_lines.clear();
        m_expr_line_count = 0;
        
        InputReaderSP reader_sp (new InputReader(interpreter.GetDebugger()));
        if (reader_sp)
        {
            Error err (reader_sp->Initialize (CommandObjectExpression::MultiLineExpressionCallback,
                                              this,                         // baton
                                              eInputReaderGranularityLine,  // token size, to pass to callback function
                                              NULL,                         // end token
                                              NULL,                         // prompt
                                              true));                       // echo input
            if (err.Success())
            {
                interpreter.GetDebugger().PushInputReader (reader_sp);
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
        return result.Succeeded();
    }

    if (command[0] == '-')
    {
        // We have some options and these options MUST end with --.
        const char *end_options = NULL;
        const char *s = command;
        while (s && s[0])
        {
            end_options = ::strstr (s, "--");
            if (end_options)
            {
                end_options += 2; // Get past the "--"
                if (::isspace (end_options[0]))
                {
                    expr = end_options;
                    while (::isspace (*expr))
                        ++expr;
                    break;
                }
            }
            s = end_options;
        }

        if (end_options)
        {
            Args args (command, end_options - command);
            if (!ParseOptions (interpreter, args, result))
                return false;
        }
    }

    if (expr == NULL)
        expr = command;
    
    if (EvaluateExpression (expr, false, result.GetOutputStream(), result.GetErrorStream(), &result))
        return true;

    result.SetStatus (eReturnStatusFailed);
    return false;
}

lldb::OptionDefinition
CommandObjectExpression::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_ALL, false, "language",   'l', required_argument, NULL, 0, "[c|c++|objc|objc++]",          "Sets the language to use when parsing the expression."},
{ LLDB_OPT_SET_ALL, false, "format",     'f', required_argument, NULL, 0, "[ [bool|b] | [bin] | [char|c] | [oct|o] | [dec|i|d|u] | [hex|x] | [float|f] | [cstr|s] ]",  "Specify the format that the expression output should use."},
{ LLDB_OPT_SET_ALL, false, "debug",      'g', no_argument,       NULL, 0, NULL,                           "Enable verbose debug logging of the expression parsing and evaluation."},
{ LLDB_OPT_SET_ALL, false, "use-ir",     'i', no_argument,       NULL, 0, NULL,                           "[Temporary] Instructs the expression evaluator to use IR instead of ASTs."},
{ 0, false, NULL, 0, 0, NULL, NULL, NULL, NULL }
};

