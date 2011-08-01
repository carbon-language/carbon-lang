//===-- ClangUserExpression.cpp ---------------------------------*- C++ -*-===//
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
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/ASTResultSynthesizer.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/ClangExpressionParser.h"
#include "lldb/Expression/ClangFunction.h"
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
    ClangExpression (),
    m_expr_text (expr),
    m_expr_prefix (expr_prefix ? expr_prefix : ""),
    m_transformed_text (),
    m_desired_type (NULL, NULL),
    m_cplusplus (false),
    m_objectivec (false),
    m_needs_object_ptr (false),
    m_const_object (false),
    m_const_result ()
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
        
        if (ClangASTContext::IsPointerType(this_type->GetClangForwardType(),
                                           &pointer_target_type))
        {
            TypeFromUser target_ast_type(pointer_target_type, this_type->GetClangAST());
            
            if (ClangASTContext::IsCXXClassType(target_ast_type.GetOpaqueQualType()))
            {
                m_cplusplus = true;
            
                if (target_ast_type.IsConst())
                    m_const_object = true;
            }
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
                            TypeFromUser desired_type,
                            bool keep_result_in_memory)
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
    else if (m_objectivec)
    {
        const char *function_name = FunctionName();
        
        m_transformed_stream.Printf("%s                                                     \n"
                                    "typedef unsigned short unichar;                        \n"
                                    "@interface $__lldb_objc_class ($__lldb_category)       \n"
                                    "-(void)%s:(void *)$__lldb_arg;                         \n"
                                    "@end                                                   \n"
                                    "@implementation $__lldb_objc_class ($__lldb_category)  \n"
                                    "-(void)%s:(void *)$__lldb_arg                          \n"
                                    "{                                                      \n"
                                    "    %s;                                                \n"
                                    "}                                                      \n"
                                    "@end                                                   \n",
                                    m_expr_prefix.c_str(),
                                    function_name,
                                    function_name,
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
    
    //////////////////////////
    // Parse the expression
    //
    
    m_desired_type = desired_type;
    
    m_expr_decl_map.reset(new ClangExpressionDeclMap(keep_result_in_memory));
    
    if (!m_expr_decl_map->WillParse(exe_ctx))
    {
        error_stream.PutCString ("error: current process state is unsuitable for expression parsing\n");
        return false;
    }
    
    ClangExpressionParser parser(exe_ctx.process, *this);
    
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
    m_dwarf_opcodes->SetByteOrder (lldb::endian::InlHostByteOrder());
    m_dwarf_opcodes->GetFlags ().Set (Stream::eBinary);
    
    m_local_variables.reset(new ClangExpressionVariableList());
            
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
    
    m_data_allocator.reset(new ProcessDataAllocator(*exe_ctx.process));
    
    Error jit_error = parser.MakeJIT (m_jit_alloc, m_jit_start_addr, m_jit_end_addr, exe_ctx, m_data_allocator.get(), m_const_result, true);
    
    if (log)
    {
        StreamString dump_string;
        m_data_allocator->Dump(dump_string);
        
        log->Printf("Data buffer contents:\n%s", dump_string.GetString().c_str());
    }
    
    m_expr_decl_map->DidParse();
    
    if (jit_error.Success())
    {
        if (exe_ctx.process && m_jit_alloc != LLDB_INVALID_ADDRESS)
            m_jit_process_sp = exe_ctx.process->GetSP();        
        return true;
    }
    else
    {
        const char *error_cstr = jit_error.AsCString();
        if (error_cstr && error_cstr[0])
            error_stream.Printf ("error: %s\n", error_cstr);
        else
            error_stream.Printf ("error: expression can't be interpreted or run\n", num_errors);
        return false;
    }
}

bool
ClangUserExpression::PrepareToExecuteJITExpression (Stream &error_stream,
                                                    ExecutionContext &exe_ctx,
                                                    lldb::addr_t &struct_address,
                                                    lldb::addr_t &object_ptr,
                                                    lldb::addr_t &cmd_ptr)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    if (m_jit_start_addr != LLDB_INVALID_ADDRESS)
    {
        Error materialize_error;
        
        if (m_needs_object_ptr)
        {
            ConstString object_name;
            
            if (m_cplusplus)
            {
                object_name.SetCString("this");
            }
            else if (m_objectivec)
            {
                object_name.SetCString("self");
            }
            else
            {
                error_stream.Printf("Need object pointer but don't know the language\n");
                return false;
            }
            
            if (!(m_expr_decl_map->GetObjectPointer(object_ptr, object_name, exe_ctx, materialize_error)))
            {
                error_stream.Printf("Couldn't get required object pointer: %s\n", materialize_error.AsCString());
                return false;
            }
            
            if (m_objectivec)
            {
                ConstString cmd_name("_cmd");
                
                if (!(m_expr_decl_map->GetObjectPointer(cmd_ptr, cmd_name, exe_ctx, materialize_error, true)))
                {
                    error_stream.Printf("Couldn't get required object pointer: %s\n", materialize_error.AsCString());
                    return false;
                }
            }
        }
                
        if (!m_expr_decl_map->Materialize(exe_ctx, struct_address, materialize_error))
        {
            error_stream.Printf("Couldn't materialize struct: %s\n", materialize_error.AsCString());
            return false;
        }

#if 0
		// jingham: look here
        StreamFile logfile ("/tmp/exprs.txt", "a");
        logfile.Printf("0x%16.16llx: thread = 0x%4.4x, expr = '%s'\n", m_jit_start_addr, exe_ctx.thread ? exe_ctx.thread->GetID() : -1, m_expr_text.c_str());
#endif
        
        if (log)
        {
            log->Printf("-- [ClangUserExpression::PrepareToExecuteJITExpression] Materializing for execution --");
            
            log->Printf("  Function address  : 0x%llx", (uint64_t)m_jit_start_addr);
            
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
            
    lldb::addr_t object_ptr = 0;
    lldb::addr_t cmd_ptr = 0;
    
    PrepareToExecuteJITExpression (error_stream, exe_ctx, struct_address, object_ptr, cmd_ptr);
    
    // FIXME: This should really return a ThreadPlanCallUserExpression, in order to make sure that we don't release the
    // ClangUserExpression resources before the thread plan finishes execution in the target.  But because we are 
    // forcing unwind_on_error to be true here, in practical terms that can't happen.
    
    return ClangFunction::GetThreadPlanToCallFunction (exe_ctx, 
                                                       m_jit_start_addr, 
                                                       struct_address, 
                                                       error_stream,
                                                       true,
                                                       true, 
                                                       (m_needs_object_ptr ? &object_ptr : NULL),
                                                       (m_needs_object_ptr && m_objectivec) ? &cmd_ptr : NULL);
}

bool
ClangUserExpression::FinalizeJITExecution (Stream &error_stream,
                                           ExecutionContext &exe_ctx,
                                           lldb::ClangExpressionVariableSP &result,
                                           lldb::addr_t function_stack_pointer)
{
    Error expr_error;
    
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
    {
        log->Printf("-- [ClangUserExpression::FinalizeJITExecution] Dematerializing after execution --");
    
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
    
    lldb::addr_t function_stack_bottom = function_stack_pointer - Host::GetPageSize();
    
        
    if (!m_expr_decl_map->Dematerialize(exe_ctx, result, function_stack_pointer, function_stack_bottom, expr_error))
    {
        error_stream.Printf ("Couldn't dematerialize struct : %s\n", expr_error.AsCString("unknown error"));
        return false;
    }
    return true;
}        

ExecutionResults
ClangUserExpression::Execute (Stream &error_stream,
                              ExecutionContext &exe_ctx,
                              bool discard_on_error,
                              ClangUserExpression::ClangUserExpressionSP &shared_ptr_to_me,
                              lldb::ClangExpressionVariableSP &result)
{
    // The expression log is quite verbose, and if you're just tracking the execution of the
    // expression, it's quite convenient to have these logs come out with the STEP log as well.
    lldb::LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EXPRESSIONS | LIBLLDB_LOG_STEP));

    if (m_dwarf_opcodes.get())
    {
        // TODO execute the JITted opcodes
        
        error_stream.Printf("We don't currently support executing DWARF expressions");
        
        return eExecutionSetupError;
    }
    else if (m_jit_start_addr != LLDB_INVALID_ADDRESS)
    {
        lldb::addr_t struct_address;
                
        lldb::addr_t object_ptr = 0;
        lldb::addr_t cmd_ptr = 0;
        
        if (!PrepareToExecuteJITExpression (error_stream, exe_ctx, struct_address, object_ptr, cmd_ptr))
            return eExecutionSetupError;
        
        const bool stop_others = true;
        const bool try_all_threads = true;
        
        Address wrapper_address (NULL, m_jit_start_addr);
        lldb::ThreadPlanSP call_plan_sp(new ThreadPlanCallUserExpression (*(exe_ctx.thread), 
                                                                          wrapper_address, 
                                                                          struct_address, 
                                                                          stop_others, 
                                                                          discard_on_error, 
                                                                          (m_needs_object_ptr ? &object_ptr : NULL),
                                                                          ((m_needs_object_ptr && m_objectivec) ? &cmd_ptr : NULL),
                                                                          shared_ptr_to_me));
        
        if (call_plan_sp == NULL || !call_plan_sp->ValidatePlan (NULL))
            return eExecutionSetupError;
        
        lldb::addr_t function_stack_pointer = static_cast<ThreadPlanCallFunction *>(call_plan_sp.get())->GetFunctionStackPointer();
    
        call_plan_sp->SetPrivate(true);
    
        uint32_t single_thread_timeout_usec = 500000;
        
        if (log)
            log->Printf("-- [ClangUserExpression::Execute] Execution of expression begins --");
        
        ExecutionResults execution_result = exe_ctx.process->RunThreadPlan (exe_ctx, 
                                                                            call_plan_sp, 
                                                                            stop_others, 
                                                                            try_all_threads, 
                                                                            discard_on_error,
                                                                            single_thread_timeout_usec, 
                                                                            error_stream);
        
        if (log)
            log->Printf("-- [ClangUserExpression::Execute] Execution of expression completed --");

        if (execution_result == eExecutionInterrupted)
        {
            const char *error_desc = NULL;
            
            if (call_plan_sp)
            {
                lldb::StopInfoSP real_stop_info_sp = call_plan_sp->GetRealStopInfo();
                if (real_stop_info_sp)
                    error_desc = real_stop_info_sp->GetDescription();
            }
            if (error_desc)
                error_stream.Printf ("Execution was interrupted, reason: %s.", error_desc);
            else
                error_stream.Printf ("Execution was interrupted.", error_desc);
                
            if (discard_on_error)
                error_stream.Printf ("\nThe process has been returned to the state before execution.");
            else
                error_stream.Printf ("\nThe process has been left at the point where it was interrupted.");

            return execution_result;
        }
        else if (execution_result != eExecutionCompleted)
        {
            error_stream.Printf ("Couldn't execute function; result was %s\n", Process::ExecutionResultAsCString (execution_result));
            return execution_result;
        }
        
        if  (FinalizeJITExecution (error_stream, exe_ctx, result, function_stack_pointer))
            return eExecutionCompleted;
        else
            return eExecutionSetupError;
    }
    else
    {
        error_stream.Printf("Expression can't be run; neither DWARF nor a JIT compiled function is present");
        return eExecutionSetupError;
    }
}

StreamString &
ClangUserExpression::DwarfOpcodeStream ()
{
    if (!m_dwarf_opcodes.get())
        m_dwarf_opcodes.reset(new StreamString());
    
    return *m_dwarf_opcodes.get();
}

ExecutionResults
ClangUserExpression::Evaluate (ExecutionContext &exe_ctx, 
                               bool discard_on_error,
                               const char *expr_cstr,
                               const char *expr_prefix,
                               lldb::ValueObjectSP &result_valobj_sp)
{
    lldb::LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EXPRESSIONS | LIBLLDB_LOG_STEP));

    Error error;
    ExecutionResults execution_results = eExecutionSetupError;
    
    if (exe_ctx.process == NULL || exe_ctx.process->GetState() != lldb::eStateStopped)
    {
        error.SetErrorString ("must have a stopped process to evaluate expressions.");
            
        result_valobj_sp = ValueObjectConstResult::Create (NULL, error);
        return eExecutionSetupError;
    }
    
    if (!exe_ctx.process->GetDynamicCheckers())
    {
        if (log)
            log->Printf("== [ClangUserExpression::Evaluate] Installing dynamic checkers ==");
        
        DynamicCheckerFunctions *dynamic_checkers = new DynamicCheckerFunctions();
        
        StreamString install_errors;
        
        if (!dynamic_checkers->Install(install_errors, exe_ctx))
        {
            if (install_errors.GetString().empty())
                error.SetErrorString ("couldn't install checkers, unknown error");
            else
                error.SetErrorString (install_errors.GetString().c_str());
            
            result_valobj_sp = ValueObjectConstResult::Create (NULL, error);
            return eExecutionSetupError;
        }
            
        exe_ctx.process->SetDynamicCheckers(dynamic_checkers);
        
        if (log)
            log->Printf("== [ClangUserExpression::Evaluate] Finished installing dynamic checkers ==");
    }
    
    ClangUserExpressionSP user_expression_sp (new ClangUserExpression (expr_cstr, expr_prefix));

    StreamString error_stream;
        
    if (log)
        log->Printf("== [ClangUserExpression::Evaluate] Parsing expression %s ==", expr_cstr);
    
    if (!user_expression_sp->Parse (error_stream, exe_ctx, TypeFromUser(NULL, NULL), true))
    {
        if (error_stream.GetString().empty())
            error.SetErrorString ("expression failed to parse, unknown error");
        else
            error.SetErrorString (error_stream.GetString().c_str());
    }
    else
    {
        lldb::ClangExpressionVariableSP expr_result;

        if (user_expression_sp->m_const_result.get())
        {
            if (log)
                log->Printf("== [ClangUserExpression::Evaluate] Expression evaluated as a constant ==");
            
            result_valobj_sp = user_expression_sp->m_const_result->GetValueObject();
        }
        else
        {    
            error_stream.GetString().clear();
            
            if (log)
                log->Printf("== [ClangUserExpression::Evaluate] Executing expression ==");

            execution_results = user_expression_sp->Execute (error_stream, 
                                                             exe_ctx, 
                                                             discard_on_error,
                                                             user_expression_sp, 
                                                             expr_result);
            
            if (execution_results != eExecutionCompleted)
            {
                if (log)
                    log->Printf("== [ClangUserExpression::Evaluate] Execution completed abnormally ==");
                
                if (error_stream.GetString().empty())
                    error.SetErrorString ("expression failed to execute, unknown error");
                else
                    error.SetErrorString (error_stream.GetString().c_str());
            }
            else 
            {
                if (expr_result)
                {
                    result_valobj_sp = expr_result->GetValueObject();
                    
                    if (log)
                        log->Printf("== [ClangUserExpression::Evaluate] Execution completed normally with result %s ==", result_valobj_sp->GetValueAsCString());
                }
                else
                {
                    if (log)
                        log->Printf("== [ClangUserExpression::Evaluate] Execution completed normally with no result ==");
                    
                    error.SetErrorString ("Expression did not return a result");
                }
            }
        }
    }
    
    if (result_valobj_sp.get() == NULL)
        result_valobj_sp = ValueObjectConstResult::Create (NULL, error);

    return execution_results;
}
