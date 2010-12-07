//===-- ClangUserExpression.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <stdio.h>
#if HAVE_SYS_TYPES_H
#  include <sys/types.h>
#endif

// C++ Includes
#include <cstdlib>
#include <string>
#include <map>

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/ClangExpressionParser.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ASTResultSynthesizer.h"
#include "lldb/Expression/ClangUserExpression.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallUserExpression.h"

using namespace lldb_private;

ClangUserExpression::ClangUserExpression (const char *expr,
                                          const char *expr_prefix) :
    m_expr_text(expr),
    m_expr_prefix(expr_prefix ? expr_prefix : ""),
    m_transformed_text(),
    m_jit_addr(LLDB_INVALID_ADDRESS),
    m_cplusplus(false),
    m_objectivec(false),
    m_needs_object_ptr(false),
    m_const_object(false),
    m_desired_type(NULL, NULL)
{
}

ClangUserExpression::~ClangUserExpression ()
{
}

clang::ASTConsumer *
ClangUserExpression::ASTTransformer (clang::ASTConsumer *passthrough)
{
    return new ASTResultSynthesizer(passthrough,
                                    m_desired_type);
}

void
ClangUserExpression::ScanContext(ExecutionContext &exe_ctx)
{
    if (!exe_ctx.frame)
        return;
    
    VariableList *vars = exe_ctx.frame->GetVariableList(false);
    
    if (!vars)
        return;
    
    lldb::VariableSP this_var(vars->FindVariable(ConstString("this")));
    lldb::VariableSP self_var(vars->FindVariable(ConstString("self")));
    
    if (this_var.get())
    {
        Type *this_type = this_var->GetType();
        
        lldb::clang_type_t pointer_target_type;
        
        if (ClangASTContext::IsPointerType(this_type->GetClangType(),
                                           &pointer_target_type))
        {
            TypeFromUser target_ast_type(pointer_target_type, this_type->GetClangAST());
            
            if (target_ast_type.IsDefined())
                m_cplusplus = true;
            
            if (target_ast_type.IsConst())
                m_const_object = true;
        }
    }
    else if (self_var.get())
    {
        m_objectivec = true;
    }
}

// This is a really nasty hack, meant to fix Objective-C expressions of the form
// (int)[myArray count].  Right now, because the type information for count is
// not available, [myArray count] returns id, which can't be directly cast to
// int without causing a clang error.
static void
ApplyObjcCastHack(std::string &expr)
{
#define OBJC_CAST_HACK_FROM "(int)["
#define OBJC_CAST_HACK_TO   "(int)(long long)["

    size_t from_offset;
    
    while ((from_offset = expr.find(OBJC_CAST_HACK_FROM)) != expr.npos)
        expr.replace(from_offset, sizeof(OBJC_CAST_HACK_FROM) - 1, OBJC_CAST_HACK_TO);

#undef OBJC_CAST_HACK_TO
#undef OBJC_CAST_HACK_FROM
}

// Another hack, meant to allow use of unichar despite it not being available in
// the type information.  Although we could special-case it in type lookup,
// hopefully we'll figure out a way to #include the same environment as is
// present in the original source file rather than try to hack specific type
// definitions in as needed.
static void
ApplyUnicharHack(std::string &expr)
{
#define UNICHAR_HACK_FROM "unichar"
#define UNICHAR_HACK_TO   "unsigned short"
    
    size_t from_offset;
    
    while ((from_offset = expr.find(UNICHAR_HACK_FROM)) != expr.npos)
        expr.replace(from_offset, sizeof(UNICHAR_HACK_FROM) - 1, UNICHAR_HACK_TO);
    
#undef UNICHAR_HACK_TO
#undef UNICHAR_HACK_FROM
}

bool
ClangUserExpression::Parse (Stream &error_stream, 
                            ExecutionContext &exe_ctx,
                            TypeFromUser desired_type)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    ScanContext(exe_ctx);
    
    StreamString m_transformed_stream;
    
    ////////////////////////////////////
    // Generate the expression
    //
    
    ApplyObjcCastHack(m_expr_text);
    //ApplyUnicharHack(m_expr_text);

    if (m_cplusplus)
    {
        m_transformed_stream.Printf("%s                                     \n"
                                    "typedef unsigned short unichar;        \n"
                                    "void                                   \n"
                                    "$__lldb_class::%s(void *$__lldb_arg) %s\n"
                                    "{                                      \n"
                                    "    %s;                                \n" 
                                    "}                                      \n",
                                    m_expr_prefix.c_str(),
                                    FunctionName(),
                                    (m_const_object ? "const" : ""),
                                    m_expr_text.c_str());
        
        m_needs_object_ptr = true;
    }
    else
    {
        m_transformed_stream.Printf("%s                             \n"
                                    "typedef unsigned short unichar;\n"
                                    "void                           \n"
                                    "%s(void *$__lldb_arg)          \n"
                                    "{                              \n"
                                    "    %s;                        \n" 
                                    "}                              \n",
                                    m_expr_prefix.c_str(),
                                    FunctionName(),
                                    m_expr_text.c_str());
    }
    
    m_transformed_text = m_transformed_stream.GetData();
    
    
    if (log)
        log->Printf("Parsing the following code:\n%s", m_transformed_text.c_str());
    
    ////////////////////////////////////
    // Set up the target and compiler
    //
    
    Target *target = exe_ctx.target;
    
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
        
    //////////////////////////
    // Parse the expression
    //
    
    m_desired_type = desired_type;
    
    m_expr_decl_map.reset(new ClangExpressionDeclMap());
    
    m_expr_decl_map->WillParse(exe_ctx);
    
    ClangExpressionParser parser(target_triple.GetCString(), *this);
    
    unsigned num_errors = parser.Parse (error_stream);
    
    if (num_errors)
    {
        error_stream.Printf ("error: %d errors parsing expression\n", num_errors);
        
        m_expr_decl_map->DidParse();
        
        return false;
    }
    
    ///////////////////////////////////////////////
    // Convert the output of the parser to DWARF
    //

    m_dwarf_opcodes.reset(new StreamString);
    m_dwarf_opcodes->SetByteOrder (lldb::eByteOrderHost);
    m_dwarf_opcodes->GetFlags ().Set (Stream::eBinary);
    
    m_local_variables.reset(new ClangExpressionVariableStore());
            
    Error dwarf_error = parser.MakeDWARF ();
    
    if (dwarf_error.Success())
    {
        if (log)
            log->Printf("Code can be interpreted.");
        
        m_expr_decl_map->DidParse();
        
        return true;
    }
    
    //////////////////////////////////
    // JIT the output of the parser
    //
    
    m_dwarf_opcodes.reset();
    
    lldb::addr_t jit_end;
    
    Error jit_error = parser.MakeJIT (m_jit_addr, jit_end, exe_ctx);
    
    m_expr_decl_map->DidParse();
    
    if (jit_error.Success())
    {
        return true;
    }
    else
    {
        error_stream.Printf ("error: expression can't be interpreted or run\n", num_errors);
        return false;
    }
}

bool
ClangUserExpression::PrepareToExecuteJITExpression (Stream &error_stream,
                                                    ExecutionContext &exe_ctx,
                                                    lldb::addr_t &struct_address,
                                                    lldb::addr_t &object_ptr)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    if (m_jit_addr != LLDB_INVALID_ADDRESS)
    {
        
        Error materialize_error;
        
        
        if (m_needs_object_ptr && !(m_expr_decl_map->GetObjectPointer(object_ptr, exe_ctx, materialize_error)))
        {
            error_stream.Printf("Couldn't get required object pointer: %s\n", materialize_error.AsCString());
            return false;
        }
                
        if (!m_expr_decl_map->Materialize(exe_ctx, struct_address, materialize_error))
        {
            error_stream.Printf("Couldn't materialize struct: %s\n", materialize_error.AsCString());
            return false;
        }
        
        if (log)
        {
            log->Printf("-- Materializing for execution --");
            
            log->Printf("  Function address  : 0x%llx", (uint64_t)m_jit_addr);
            
            if (m_needs_object_ptr)
                log->Printf("  Object pointer    : 0x%llx", (uint64_t)object_ptr);
            
            log->Printf("  Structure address : 0x%llx", (uint64_t)struct_address);
                    
            StreamString args;
            
            Error dump_error;
            
            if (struct_address)
            {
                if (!m_expr_decl_map->DumpMaterializedStruct(exe_ctx, args, dump_error))
                {
                    log->Printf("  Couldn't extract variable values : %s", dump_error.AsCString("unknown error"));
                }
                else
                {
                    log->Printf("  Structure contents:\n%s", args.GetData());
                }
            }
        }
    }
    return true;
}

ThreadPlan *
ClangUserExpression::GetThreadPlanToExecuteJITExpression (Stream &error_stream,
                                       ExecutionContext &exe_ctx)
{
    lldb::addr_t struct_address;
            
    lldb::addr_t object_ptr = NULL;
    
    PrepareToExecuteJITExpression (error_stream, exe_ctx, struct_address, object_ptr);
    
    // FIXME: This should really return a ThreadPlanCallUserExpression, in order to make sure that we don't release the
    // ClangUserExpression resources before the thread plan finishes execution in the target.  But because we are 
    // forcing unwind_on_error to be true here, in practical terms that can't happen.  
    return ClangFunction::GetThreadPlanToCallFunction (exe_ctx, 
                                                       m_jit_addr, 
                                                       struct_address, 
                                                       error_stream,
                                                       true,
                                                       true, 
                                                       (m_needs_object_ptr ? &object_ptr : NULL));
}

bool
ClangUserExpression::FinalizeJITExecution (Stream &error_stream,
                                           ExecutionContext &exe_ctx,
                                           ClangExpressionVariable *&result)
{
    Error expr_error;
    
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
    {
        log->Printf("-- Dematerializing after execution --");
    
        StreamString args;
        
        Error dump_error;
        
        if (!m_expr_decl_map->DumpMaterializedStruct(exe_ctx, args, dump_error))
        {
            log->Printf("  Couldn't extract variable values : %s", dump_error.AsCString("unknown error"));
        }
        else
        {
            log->Printf("  Structure contents:\n%s", args.GetData());
        }
    }
        
    if (!m_expr_decl_map->Dematerialize(exe_ctx, result, expr_error))
    {
        error_stream.Printf ("Couldn't dematerialize struct : %s\n", expr_error.AsCString("unknown error"));
        return false;
    }
    return true;
}        

Process::ExecutionResults
ClangUserExpression::Execute (Stream &error_stream,
                              ExecutionContext &exe_ctx,
                              bool discard_on_error,
                              ClangUserExpression::ClangUserExpressionSP &shared_ptr_to_me,
                              ClangExpressionVariable *&result)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    if (m_dwarf_opcodes.get())
    {
        // TODO execute the JITted opcodes
        
        error_stream.Printf("We don't currently support executing DWARF expressions");
        
        return Process::eExecutionSetupError;
    }
    else if (m_jit_addr != LLDB_INVALID_ADDRESS)
    {
        lldb::addr_t struct_address;
                
        lldb::addr_t object_ptr = NULL;
        
        PrepareToExecuteJITExpression (error_stream, exe_ctx, struct_address, object_ptr);
        
        const bool stop_others = true;
        const bool try_all_threads = true;
        
        Address wrapper_address (NULL, m_jit_addr);
        lldb::ThreadPlanSP call_plan_sp(new ThreadPlanCallUserExpression (*(exe_ctx.thread), wrapper_address, struct_address, 
                                                                               stop_others, discard_on_error, 
                                                                               (m_needs_object_ptr ? &object_ptr : NULL),
                                                                               shared_ptr_to_me));
        if (call_plan_sp == NULL || !call_plan_sp->ValidatePlan (NULL))
            return Process::eExecutionSetupError;
    
        call_plan_sp->SetPrivate(true);
    
        uint32_t single_thread_timeout_usec = 10000000;
        
        if (log)
            log->Printf("-- Execution of expression begins --");
        
        Process::ExecutionResults execution_result = 
           exe_ctx.process->RunThreadPlan (exe_ctx, call_plan_sp, stop_others, try_all_threads, discard_on_error,
                                           single_thread_timeout_usec, error_stream);
        
        if (log)
            log->Printf("-- Execution of expression completed --");

        if (execution_result == Process::eExecutionInterrupted)
        {
            if (discard_on_error)
                error_stream.Printf ("Expression execution was interrupted.  The process has been returned to the state before execution.");
            else
                error_stream.Printf ("Expression execution was interrupted.  The process has been left at the point where it was interrupted.");

            return execution_result;
        }
        else if (execution_result != Process::eExecutionCompleted)
        {
            error_stream.Printf ("Couldn't execute function; result was %s\n", Process::ExecutionResultAsCString (execution_result));
            return execution_result;
        }
        
        if  (FinalizeJITExecution (error_stream, exe_ctx, result))
            return Process::eExecutionCompleted;
        else
            return Process::eExecutionSetupError;
    }
    else
    {
        error_stream.Printf("Expression can't be run; neither DWARF nor a JIT compiled function is present");
        return Process::eExecutionSetupError;
    }
}

StreamString &
ClangUserExpression::DwarfOpcodeStream ()
{
    if (!m_dwarf_opcodes.get())
        m_dwarf_opcodes.reset(new StreamString());
    
    return *m_dwarf_opcodes.get();
}

Process::ExecutionResults
ClangUserExpression::Evaluate (ExecutionContext &exe_ctx, 
                               bool discard_on_error,
                               const char *expr_cstr,
                               const char *expr_prefix,
                               lldb::ValueObjectSP &result_valobj_sp)
{
    Error error;
    Process::ExecutionResults execution_results = Process::eExecutionSetupError;
    
    if (exe_ctx.process == NULL)
    {
        error.SetErrorString ("Must have a process to evaluate expressions.");
            
        result_valobj_sp.reset (new ValueObjectConstResult (error));
        return Process::eExecutionSetupError;
    }
    
    if (!exe_ctx.process->GetDynamicCheckers())
    {
        DynamicCheckerFunctions *dynamic_checkers = new DynamicCheckerFunctions();
        
        StreamString install_errors;
        
        if (!dynamic_checkers->Install(install_errors, exe_ctx))
        {
            if (install_errors.GetString().empty())
                error.SetErrorString ("couldn't install checkers, unknown error");
            else
                error.SetErrorString (install_errors.GetString().c_str());
            
            result_valobj_sp.reset (new ValueObjectConstResult (error));
            return Process::eExecutionSetupError;
        }
            
        exe_ctx.process->SetDynamicCheckers(dynamic_checkers);
    }
    
    ClangUserExpressionSP user_expression_sp (new ClangUserExpression (expr_cstr, expr_prefix));

    StreamString error_stream;
    
    if (!user_expression_sp->Parse (error_stream, exe_ctx, TypeFromUser(NULL, NULL)))
    {
        if (error_stream.GetString().empty())
            error.SetErrorString ("expression failed to parse, unknown error");
        else
            error.SetErrorString (error_stream.GetString().c_str());
    }
    else
    {
        ClangExpressionVariable *expr_result = NULL;

        error_stream.GetString().clear();

        execution_results = user_expression_sp->Execute (error_stream, 
                                                         exe_ctx, 
                                                         discard_on_error, 
                                                         user_expression_sp, 
                                                         expr_result);
        if (execution_results != Process::eExecutionCompleted)
        {
            if (error_stream.GetString().empty())
                error.SetErrorString ("expression failed to execute, unknown error");
            else
                error.SetErrorString (error_stream.GetString().c_str());
        }
        else 
        {
            // TODO: seems weird to get a pointer to a result object back from
            // a function. Do we own it? Feels like we do, but from looking at the
            // code we don't. Might be best to make this a reference and state
            // explicitly that we don't own it when we get a reference back from
            // the execute?
            if (expr_result)
            {
                result_valobj_sp = expr_result->GetExpressionResult (&exe_ctx);
            }
            else
            {
                error.SetErrorString ("Expression did not return a result");
            }
        }
    }
    
    if (result_valobj_sp.get() == NULL)
        result_valobj_sp.reset (new ValueObjectConstResult (error));

    return execution_results;
}
