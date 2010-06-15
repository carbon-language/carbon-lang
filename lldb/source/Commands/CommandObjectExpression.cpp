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
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/ClangExpressionVariable.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/CommandContext.h"
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
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    return false;
}


size_t
CommandObjectExpression::MultiLineExpressionCallback
(
    void *baton, 
    InputReader *reader, 
    lldb::InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    FILE *out_fh = Debugger::GetSharedInstance().GetOutputFileHandle();
    CommandObjectExpression *cmd_object_expr = (CommandObjectExpression *) baton;

    switch (notification)
    {
    case eInputReaderActivate:
        if (out_fh)
            ::fprintf (out_fh, "%s\n", "Enter expressions, then terminate with an empty line to evaluate:");
        // Fall through
    case eInputReaderReactivate:
        //if (out_fh)
        //    ::fprintf (out_fh, "%3u: ", cmd_object_expr->m_expr_line_count);
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
            reader->SetIsDone(true);
        //else if (out_fh && !reader->IsDone())
        //    ::fprintf (out_fh, "%3u: ", cmd_object_expr->m_expr_line_count);
        break;
        
    case eInputReaderDone:
        {
            StreamFile out_stream(Debugger::GetSharedInstance().GetOutputFileHandle());
            StreamFile err_stream(Debugger::GetSharedInstance().GetErrorFileHandle());
            bool bare = false;
            cmd_object_expr->EvaluateExpression (cmd_object_expr->m_expr_lines.c_str(), 
                                                 bare, 
                                                 out_stream, 
                                                 err_stream);
        }
        break;
    }

    return bytes_len;
}

bool
CommandObjectExpression::EvaluateExpression (const char *expr, bool bare, Stream &output_stream, Stream &error_stream)
{
    bool success = false;
    ConstString target_triple;
    Target *target = m_exe_ctx.target;
    if (target)
        target->GetTargetTriple(target_triple);

    if (!target_triple)
        target_triple = Host::GetTargetTriple ();


    if (target_triple)
    {
        const bool show_types = m_options.show_types;
        const bool show_summary = m_options.show_summary;
        const bool debug = m_options.debug;
        
        ClangExpressionDeclMap expr_decl_map(&m_exe_ctx);
        ClangExpression clang_expr(target_triple.AsCString(), &expr_decl_map);
        
        unsigned num_errors = 0;
        
        if (bare)
            num_errors = clang_expr.ParseBareExpression (llvm::StringRef(expr), error_stream);
        else
            num_errors = clang_expr.ParseExpression (expr, error_stream);

        if (num_errors == 0)
        {
            StreamString dwarf_opcodes;
            dwarf_opcodes.SetByteOrder(eByteOrderHost);
            dwarf_opcodes.GetFlags().Set(Stream::eBinary);
            ClangExpressionVariableList expr_local_vars;
            clang_expr.ConvertExpressionToDWARF (expr_local_vars, dwarf_opcodes);

            success = true;

            DataExtractor dwarf_opcodes_data(dwarf_opcodes.GetData(), dwarf_opcodes.GetSize(), eByteOrderHost, 8);
            DWARFExpression expr(dwarf_opcodes_data, 0, dwarf_opcodes_data.GetByteSize(), NULL);
            expr.SetExpressionLocalVariableList(&expr_local_vars);
            if (debug)
            {
                output_stream << "Expression parsed ok, dwarf opcodes:";
                output_stream.IndentMore();
                expr.GetDescription(&output_stream, lldb::eDescriptionLevelVerbose);
                output_stream.IndentLess();
                output_stream.EOL();
            }

            clang::ASTContext *ast_context = clang_expr.GetASTContext();
            Value expr_result;
            Error expr_error;
            bool expr_success = expr.Evaluate (&m_exe_ctx, ast_context, NULL, expr_result, &expr_error);
            if (expr_success)
            {
                lldb::Format format = m_options.format;

                // Resolve any values that are possible
                expr_result.ResolveValue(&m_exe_ctx, ast_context);

                if (expr_result.GetContextType() == Value::eContextTypeInvalid &&
                    expr_result.GetValueType() == Value::eValueTypeScalar &&
                    format == eFormatDefault)
                {
                    // The expression result is just a scalar with no special formatting
                    expr_result.GetScalar().GetValue (&output_stream, show_types);
                    output_stream.EOL();
                }
                else
                {
                    DataExtractor data;
                    expr_error = expr_result.GetValueAsData (&m_exe_ctx, ast_context, data, 0);
                    if (expr_error.Success())
                    {
                        if (format == eFormatDefault)
                            format = expr_result.GetValueDefaultFormat ();

                        void *clang_type = expr_result.GetValueOpaqueClangQualType();
                        if (clang_type)
                        {
                            if (show_types)
                                Type::DumpClangTypeName(&output_stream, clang_type);

                            Type::DumpValue (
                                &m_exe_ctx,                 // The execution context for memory and variable access
                                ast_context,                // The ASTContext that the clang type belongs to
                                clang_type,                 // The opaque clang type we want to dump that value of
                                &output_stream,             // Stream to dump to
                                format,                     // Format to use when dumping
                                data,                       // A buffer containing the bytes for the clang type
                                0,                          // Byte offset within "data" where value is
                                data.GetByteSize(),         // Size in bytes of the value we are dumping
                                0,                          // Bitfield bit size
                                0,                          // Bitfield bit offset
                                show_types,                 // Show types?
                                show_summary,               // Show summary?
                                debug,                      // Debug logging output?
                                UINT32_MAX);                // Depth to dump in case this is an aggregate type
                        }
                        else
                        {
                            data.Dump(&output_stream,       // Stream to dump to
                                      0,                    // Byte offset within "data"
                                      format,               // Format to use when dumping
                                      data.GetByteSize(),   // Size in bytes of each item we are dumping
                                      1,                    // Number of items to dump
                                      UINT32_MAX,           // Number of items per line
                                      LLDB_INVALID_ADDRESS,   // Invalid address, don't show any offset/address context
                                      0,                    // Bitfield bit size
                                      0);                   // Bitfield bit offset
                        }
                        output_stream.EOL();
                    }
                    else
                    {
                        error_stream.Printf ("error: %s\n", expr_error.AsCString());
                        success = false;
                    }
                }
            }
            else
            {
                error_stream.Printf ("error: %s\n", expr_error.AsCString());
            }
        }
    }
    else
    {
        error_stream.PutCString ("error: invalid target triple\n");
    }

    return success;
}

bool
CommandObjectExpression::ExecuteRawCommandString
(
    const char *command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    ConstString target_triple;
    Target *target = context->GetTarget ();
    if (target)
        target->GetTargetTriple(target_triple);

    if (!target_triple)
        target_triple = Host::GetTargetTriple ();

    ExecutionContext exe_ctx(context->GetExecutionContext());

    Stream &output_stream = result.GetOutputStream();

    m_options.ResetOptionValues();

    const char * expr = NULL;

    if (command[0] == '\0')
    {
        m_expr_lines.clear();
        m_expr_line_count = 0;
        
        InputReaderSP reader_sp (new InputReader());
        if (reader_sp)
        {
            Error err (reader_sp->Initialize (CommandObjectExpression::MultiLineExpressionCallback,
                                              this,                         // baton
                                              eInputReaderGranularityLine,  // token size, to pass to callback function
                                              NULL,                       // end token
                                              NULL,                         // prompt
                                              true));                       // echo input
            if (err.Success())
            {
                Debugger::GetSharedInstance().PushInputReader (reader_sp);
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
            Args args(command, end_options - command);
            if (!ParseOptions(args, interpreter, result))
                return false;
        }
    }

    const bool show_types = m_options.show_types;
    const bool show_summary = m_options.show_summary;
    const bool debug = m_options.debug;


    if (expr == NULL)
        expr = command;

    if (target_triple)
    {
        ClangExpressionDeclMap expr_decl_map(&exe_ctx);

        ClangExpression clang_expr(target_triple.AsCString(), &expr_decl_map);        
        
        unsigned num_errors = clang_expr.ParseExpression (expr, result.GetErrorStream());

        if (num_errors == 0)
        {
            StreamString dwarf_opcodes;
            dwarf_opcodes.SetByteOrder(eByteOrderHost);
            dwarf_opcodes.GetFlags().Set(Stream::eBinary);
            ClangExpressionVariableList expr_local_vars;
            clang_expr.ConvertExpressionToDWARF (expr_local_vars, dwarf_opcodes);

            result.SetStatus (eReturnStatusSuccessFinishResult);

            DataExtractor dwarf_opcodes_data(dwarf_opcodes.GetData(), dwarf_opcodes.GetSize(), eByteOrderHost, 8);
            DWARFExpression expr(dwarf_opcodes_data, 0, dwarf_opcodes_data.GetByteSize(), NULL);
            expr.SetExpressionLocalVariableList(&expr_local_vars);
            expr.SetExpressionDeclMap(&expr_decl_map);
            if (debug)
            {
                output_stream << "Expression parsed ok, dwarf opcodes:";
                output_stream.IndentMore();
                expr.GetDescription(&output_stream, lldb::eDescriptionLevelVerbose);
                output_stream.IndentLess();
                output_stream.EOL();
            }

            clang::ASTContext *ast_context = clang_expr.GetASTContext();
            Value expr_result;
            Error expr_error;
            bool expr_success = expr.Evaluate (&exe_ctx, ast_context, NULL, expr_result, &expr_error);
            if (expr_success)
            {
                lldb::Format format = m_options.format;

                // Resolve any values that are possible
                expr_result.ResolveValue(&exe_ctx, ast_context);

                if (expr_result.GetContextType() == Value::eContextTypeInvalid &&
                    expr_result.GetValueType() == Value::eValueTypeScalar &&
                    format == eFormatDefault)
                {
                    // The expression result is just a scalar with no special formatting
                    expr_result.GetScalar().GetValue (&output_stream, show_types);
                    output_stream.EOL();
                }
                else
                {
                    DataExtractor data;
                    expr_error = expr_result.GetValueAsData (&exe_ctx, ast_context, data, 0);
                    if (expr_error.Success())
                    {
                        if (format == eFormatDefault)
                            format = expr_result.GetValueDefaultFormat ();

                        void *clang_type = expr_result.GetValueOpaqueClangQualType();
                        if (clang_type)
                        {
                            if (show_types)
                                Type::DumpClangTypeName(&output_stream, clang_type);

                            Type::DumpValue (
                                &exe_ctx,                   // The execution context for memory and variable access
                                ast_context,                // The ASTContext that the clang type belongs to
                                clang_type,                 // The opaque clang type we want to dump that value of
                                &output_stream,             // Stream to dump to
                                format,                     // Format to use when dumping
                                data,                       // A buffer containing the bytes for the clang type
                                0,                          // Byte offset within "data" where value is
                                data.GetByteSize(),         // Size in bytes of the value we are dumping
                                0,                          // Bitfield bit size
                                0,                          // Bitfield bit offset
                                show_types,                 // Show types?
                                show_summary,               // Show summary?
                                debug,                      // Debug logging output?
                                UINT32_MAX);                // Depth to dump in case this is an aggregate type
                        }
                        else
                        {
                            data.Dump(&output_stream,       // Stream to dump to
                                      0,                    // Byte offset within "data"
                                      format,               // Format to use when dumping
                                      data.GetByteSize(),   // Size in bytes of each item we are dumping
                                      1,                    // Number of items to dump
                                      UINT32_MAX,           // Number of items per line
                                      LLDB_INVALID_ADDRESS,   // Invalid address, don't show any offset/address context
                                      0,                    // Bitfield bit size
                                      0);                   // Bitfield bit offset
                        }
                        output_stream.EOL();
                    }
                    else
                    {
                        result.AppendError(expr_error.AsCString());
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
            }
            else
            {
                result.AppendError (expr_error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.SetStatus (eReturnStatusFailed);
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
CommandObjectExpression::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, true,  "language",   'l', required_argument, NULL, 0, "[c|c++|objc|objc++]",          "Sets the language to use when parsing the expression."},
{ LLDB_OPT_SET_2, false, "format",     'f', required_argument, NULL, 0, "[ [bool|b] | [bin] | [char|c] | [oct|o] | [dec|i|d|u] | [hex|x] | [float|f] | [cstr|s] ]",  "Specify the format that the expression output should use."},
{ LLDB_OPT_SET_3, false, "debug",      'g', no_argument,       NULL, 0, NULL,                           "Enable verbose debug logging of the expression parsing and evaluation."},
{ 0, false, NULL, 0, 0, NULL, NULL, NULL, NULL }
};

