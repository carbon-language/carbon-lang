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
CommandObjectExpression::EvaluateExpression (const char *expr, bool bare, Stream &output_stream, Stream &error_stream)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    ////////////////////////////////////
    // Set up the target and compiler
    //
        
    Target *target = m_exe_ctx.target;
    
    if (!target)
    {
        error_stream.PutCString ("error: invalid target\n");
        return false;
    }
    
    ConstString target_triple;

    target->GetTargetTriple (target_triple);

    if (!target_triple)
        target_triple = Host::GetTargetTriple ();
    
    if (!target_triple)
    {
        error_stream.PutCString ("error: invalid target triple\n");
        return false;
    }
    
    ClangPersistentVariables persistent_vars; /* TODO store this somewhere sensible */
    ClangExpressionDeclMap expr_decl_map (&m_exe_ctx, persistent_vars);
    ClangExpression clang_expr (target_triple.AsCString (), &expr_decl_map);
    
    //////////////////////////
    // Parse the expression
    //
    
    unsigned num_errors;
    
    if (bare)
        num_errors = clang_expr.ParseBareExpression (llvm::StringRef (expr), error_stream);
    else
        num_errors = clang_expr.ParseExpression (expr, error_stream, true);

    if (num_errors)
    {
        error_stream.Printf ("error: %d errors parsing expression\n", num_errors);
        return false;
    }
    
    ///////////////////////////////////////////////
    // Convert the output of the parser to DWARF
    //
    
    StreamString dwarf_opcodes;
    dwarf_opcodes.SetByteOrder (eByteOrderHost);
    dwarf_opcodes.GetFlags ().Set (Stream::eBinary);
    
    ClangExpressionVariableList expr_local_vars;

    bool success;
    bool canInterpret = false;
    
    clang::ASTContext *ast_context = clang_expr.GetASTContext ();
    Value expr_result;
    Error expr_error;
    
    canInterpret = clang_expr.ConvertIRToDWARF (expr_local_vars, dwarf_opcodes);
    
    if (canInterpret)
    {
        if (log)
            log->Printf("Code can be interpreted.");
        success = true;
    }
    else
    {
        if (log)
            log->Printf("Code cannot be interpreted and must be run in the target.");
        success = clang_expr.PrepareIRForTarget (expr_local_vars);
    }
    
    if (!success)
    {
        error_stream.PutCString ("error: expression couldn't be converted to IR\n");
        return false;
    }
    
    if (canInterpret)
    {
        // TODO interpret IR
        return false;
    }
    else
    {
        if (!clang_expr.JITFunction (m_exe_ctx, "___clang_expr"))
        {
            error_stream.PutCString ("error: IR could not be JIT compiled\n");
            return false;
        }
        
        if (!clang_expr.WriteJITCode (m_exe_ctx))
        {
            error_stream.PutCString ("error: JIT code could not be written to the target\n");
            return false;
        }
        
        lldb::addr_t function_address(clang_expr.GetFunctionAddress ("___clang_expr"));
        
        if (function_address == LLDB_INVALID_ADDRESS)
        {
            error_stream.PutCString ("JIT compiled code's address couldn't be found\n");
            return false;
        }
        
        lldb::addr_t struct_address;
        
        if (!expr_decl_map.Materialize(&m_exe_ctx, struct_address, expr_error))
        {
            error_stream.Printf ("Couldn't materialize struct: %s\n", expr_error.AsCString("unknown error"));
            return false;
        }
        
        if (log)
        {
            log->Printf("Function address  : 0x%llx", (uint64_t)function_address);
            log->Printf("Structure address : 0x%llx", (uint64_t)struct_address);
            
            StreamString insns;

            Error err = clang_expr.DisassembleFunction(insns, m_exe_ctx, "___clang_expr");
            
            if (!err.Success())
            {
                log->Printf("Couldn't disassemble function : %s", err.AsCString("unknown error"));
            }
            else
            {
                log->Printf("Function disassembly:\n%s", insns.GetData());
            }
            
            StreamString args;
            
            if (!expr_decl_map.DumpMaterializedStruct(&m_exe_ctx, args, err))
            {
                log->Printf("Couldn't extract variable values : %s", err.AsCString("unknown error"));
            }
            else
            {
                log->Printf("Structure contents:\n%s", args.GetData());
            }
        }
                    
        ClangFunction::ExecutionResults execution_result = 
            ClangFunction::ExecuteFunction (m_exe_ctx, function_address, struct_address, true, true, 10000, error_stream);
        
        if (execution_result != ClangFunction::eExecutionCompleted)
        {
            const char *result_name;
            
            switch (execution_result)
            {
            case ClangFunction::eExecutionCompleted:
                result_name = "eExecutionCompleted";
                break;
            case ClangFunction::eExecutionDiscarded:
                result_name = "eExecutionDiscarded";
                break;
            case ClangFunction::eExecutionInterrupted:
                result_name = "eExecutionInterrupted";
                break;
            case ClangFunction::eExecutionSetupError:
                result_name = "eExecutionSetupError";
                break;
            case ClangFunction::eExecutionTimedOut:
                result_name = "eExecutionTimedOut";
                break;
            }
            
            error_stream.Printf ("Couldn't execute function; result was %s\n", result_name);
            return false;
        }
                    
        if (!expr_decl_map.Dematerialize(&m_exe_ctx, expr_result, expr_error))
        {
            error_stream.Printf ("Couldn't dematerialize struct: %s\n", expr_error.AsCString("unknown error"));
            return false;
        }
    }
            
    ///////////////////////////////////////
    // Interpret the result and print it
    //
    
    lldb::Format format = m_options.format;
    
    // Resolve any values that are possible
    expr_result.ResolveValue (&m_exe_ctx, ast_context);
    
    if (expr_result.GetContextType () == Value::eContextTypeInvalid &&
        expr_result.GetValueType () == Value::eValueTypeScalar &&
        format == eFormatDefault)
    {
        // The expression result is just a scalar with no special formatting
        expr_result.GetScalar ().GetValue (&output_stream, m_options.show_types);
        output_stream.EOL ();
        return true;
    }
    
    // The expression result is more complext and requires special handling
    DataExtractor data;
    expr_error = expr_result.GetValueAsData (&m_exe_ctx, ast_context, data, 0);
    
    if (!expr_error.Success ())
    {
        error_stream.Printf ("error: couldn't resolve result value: %s\n", expr_error.AsCString ());
        return false;
    }
    
    if (format == eFormatDefault)
        format = expr_result.GetValueDefaultFormat ();
    
    void *clang_type = expr_result.GetValueOpaqueClangQualType ();
    
    if (clang_type)
    {
        if (m_options.show_types)
            output_stream.Printf("(%s) ", ClangASTType::GetClangTypeName (clang_type).GetCString());
        
        ClangASTType::DumpValue (ast_context,               // The ASTContext that the clang type belongs to
                                 clang_type,                // The opaque clang type we want to dump that value of
                                 &m_exe_ctx,                // The execution context for memory and variable access
                                 &output_stream,            // Stream to dump to
                                 format,                    // Format to use when dumping
                                 data,                      // A buffer containing the bytes for the clang type
                                 0,                         // Byte offset within "data" where value is
                                 data.GetByteSize (),       // Size in bytes of the value we are dumping
                                 0,                         // Bitfield bit size
                                 0,                         // Bitfield bit offset
                                 m_options.show_types,      // Show types?
                                 m_options.show_summary,    // Show summary?
                                 m_options.debug,           // Debug logging output?
                                 UINT32_MAX);               // Depth to dump in case this is an aggregate type
    }
    else
    {
        data.Dump (&output_stream,          // Stream to dump to
                   0,                       // Byte offset within "data"
                   format,                  // Format to use when dumping
                   data.GetByteSize (),     // Size in bytes of each item we are dumping
                   1,                       // Number of items to dump
                   UINT32_MAX,              // Number of items per line
                   LLDB_INVALID_ADDRESS,    // Invalid address, don't show any offset/address context
                   0,                       // Bitfield bit size
                   0);                      // Bitfield bit offset
    }
    output_stream.EOL();
    
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
    
    return EvaluateExpression (expr, false, result.GetOutputStream(), result.GetErrorStream());
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

