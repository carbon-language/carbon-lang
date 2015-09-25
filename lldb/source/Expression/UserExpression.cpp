//===-- UserExpression.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#if HAVE_SYS_TYPES_H
#  include <sys/types.h>
#endif

#include <cstdlib>
#include <string>
#include <map>

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Expression/IRInterpreter.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Expression/UserExpression.h"
#include "Plugins/ExpressionParser/Clang/ClangPersistentVariables.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallUserExpression.h"

using namespace lldb_private;

UserExpression::UserExpression (ExecutionContextScope &exe_scope,
                                const char *expr,
                                const char *expr_prefix,
                                lldb::LanguageType language,
                                ResultType desired_type) :
    Expression (exe_scope),
    m_stack_frame_bottom (LLDB_INVALID_ADDRESS),
    m_stack_frame_top (LLDB_INVALID_ADDRESS),
    m_expr_text (expr),
    m_expr_prefix (expr_prefix ? expr_prefix : ""),
    m_language (language),
    m_transformed_text (),
    m_desired_type (desired_type),
    m_execution_unit_sp(),
    m_materializer_ap(),
    m_jit_module_wp(),
    m_enforce_valid_object (true),
    m_in_cplusplus_method (false),
    m_in_objectivec_method (false),
    m_in_static_method(false),
    m_needs_object_ptr (false),
    m_const_object (false),
    m_target (NULL),
    m_can_interpret (false),
    m_materialized_address (LLDB_INVALID_ADDRESS)
{
}

UserExpression::~UserExpression ()
{
    if (m_target)
    {
        lldb::ModuleSP jit_module_sp (m_jit_module_wp.lock());
        if (jit_module_sp)
            m_target->GetImages().Remove(jit_module_sp);
    }
}

void
UserExpression::InstallContext (ExecutionContext &exe_ctx)
{
    m_jit_process_wp = exe_ctx.GetProcessSP();

    lldb::StackFrameSP frame_sp = exe_ctx.GetFrameSP();

    if (frame_sp)
        m_address = frame_sp->GetFrameCodeAddress();
}

bool
UserExpression::LockAndCheckContext (ExecutionContext &exe_ctx,
                                          lldb::TargetSP &target_sp,
                                          lldb::ProcessSP &process_sp,
                                          lldb::StackFrameSP &frame_sp)
{
    lldb::ProcessSP expected_process_sp = m_jit_process_wp.lock();
    process_sp = exe_ctx.GetProcessSP();

    if (process_sp != expected_process_sp)
        return false;

    process_sp = exe_ctx.GetProcessSP();
    target_sp = exe_ctx.GetTargetSP();
    frame_sp = exe_ctx.GetFrameSP();

    if (m_address.IsValid())
    {
        if (!frame_sp)
            return false;
        else
            return (0 == Address::CompareLoadAddress(m_address, frame_sp->GetFrameCodeAddress(), target_sp.get()));
    }

    return true;
}

bool
UserExpression::MatchesContext (ExecutionContext &exe_ctx)
{
    lldb::TargetSP target_sp;
    lldb::ProcessSP process_sp;
    lldb::StackFrameSP frame_sp;

    return LockAndCheckContext(exe_ctx, target_sp, process_sp, frame_sp);
}

lldb::addr_t
UserExpression::GetObjectPointer (lldb::StackFrameSP frame_sp,
                  ConstString &object_name,
                  Error &err)
{
    err.Clear();

    if (!frame_sp)
    {
        err.SetErrorStringWithFormat("Couldn't load '%s' because the context is incomplete", object_name.AsCString());
        return LLDB_INVALID_ADDRESS;
    }

    lldb::VariableSP var_sp;
    lldb::ValueObjectSP valobj_sp;

    valobj_sp = frame_sp->GetValueForVariableExpressionPath(object_name.AsCString(),
                                                            lldb::eNoDynamicValues,
                                                            StackFrame::eExpressionPathOptionCheckPtrVsMember |
                                                            StackFrame::eExpressionPathOptionsNoFragileObjcIvar |
                                                            StackFrame::eExpressionPathOptionsNoSyntheticChildren |
                                                            StackFrame::eExpressionPathOptionsNoSyntheticArrayRange,
                                                            var_sp,
                                                            err);

    if (!err.Success() || !valobj_sp.get())
        return LLDB_INVALID_ADDRESS;

    lldb::addr_t ret = valobj_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

    if (ret == LLDB_INVALID_ADDRESS)
    {
        err.SetErrorStringWithFormat("Couldn't load '%s' because its value couldn't be evaluated", object_name.AsCString());
        return LLDB_INVALID_ADDRESS;
    }

    return ret;
}

bool
UserExpression::PrepareToExecuteJITExpression (Stream &error_stream,
                                                    ExecutionContext &exe_ctx,
                                                    lldb::addr_t &struct_address)
{
    lldb::TargetSP target;
    lldb::ProcessSP process;
    lldb::StackFrameSP frame;

    if (!LockAndCheckContext(exe_ctx,
                             target,
                             process,
                             frame))
    {
        error_stream.Printf("The context has changed before we could JIT the expression!\n");
        return false;
    }

    if (m_jit_start_addr != LLDB_INVALID_ADDRESS || m_can_interpret)
    {
        if (m_materialized_address == LLDB_INVALID_ADDRESS)
        {
            Error alloc_error;

            IRMemoryMap::AllocationPolicy policy = m_can_interpret ? IRMemoryMap::eAllocationPolicyHostOnly : IRMemoryMap::eAllocationPolicyMirror;

            m_materialized_address = m_execution_unit_sp->Malloc(m_materializer_ap->GetStructByteSize(),
                                                                 m_materializer_ap->GetStructAlignment(),
                                                                 lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                                                 policy,
                                                                 alloc_error);

            if (!alloc_error.Success())
            {
                error_stream.Printf("Couldn't allocate space for materialized struct: %s\n", alloc_error.AsCString());
                return false;
            }
        }

        struct_address = m_materialized_address;

        if (m_can_interpret && m_stack_frame_bottom == LLDB_INVALID_ADDRESS)
        {
            Error alloc_error;

            const size_t stack_frame_size = 512 * 1024;

            m_stack_frame_bottom = m_execution_unit_sp->Malloc(stack_frame_size,
                                                               8,
                                                               lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                                               IRMemoryMap::eAllocationPolicyHostOnly,
                                                               alloc_error);

            m_stack_frame_top = m_stack_frame_bottom + stack_frame_size;

            if (!alloc_error.Success())
            {
                error_stream.Printf("Couldn't allocate space for the stack frame: %s\n", alloc_error.AsCString());
                return false;
            }
        }

        Error materialize_error;

        m_dematerializer_sp = m_materializer_ap->Materialize(frame, *m_execution_unit_sp, struct_address, materialize_error);

        if (!materialize_error.Success())
        {
            error_stream.Printf("Couldn't materialize: %s\n", materialize_error.AsCString());
            return false;
        }
    }
    return true;
}

bool
UserExpression::FinalizeJITExecution (Stream &error_stream,
                                           ExecutionContext &exe_ctx,
                                           lldb::ExpressionVariableSP &result,
                                           lldb::addr_t function_stack_bottom,
                                           lldb::addr_t function_stack_top)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    if (log)
        log->Printf("-- [UserExpression::FinalizeJITExecution] Dematerializing after execution --");

    if (!m_dematerializer_sp)
    {
        error_stream.Printf ("Couldn't apply expression side effects : no dematerializer is present");
        return false;
    }

    Error dematerialize_error;

    m_dematerializer_sp->Dematerialize(dematerialize_error, result, function_stack_bottom, function_stack_top);

    if (!dematerialize_error.Success())
    {
        error_stream.Printf ("Couldn't apply expression side effects : %s\n", dematerialize_error.AsCString("unknown error"));
        return false;
    }

    if (result)
        result->TransferAddress();

    m_dematerializer_sp.reset();

    return true;
}

lldb::ExpressionResults
UserExpression::Execute (Stream &error_stream,
                              ExecutionContext &exe_ctx,
                              const EvaluateExpressionOptions& options,
                              lldb::UserExpressionSP &shared_ptr_to_me,
                              lldb::ExpressionVariableSP &result)
{
    // The expression log is quite verbose, and if you're just tracking the execution of the
    // expression, it's quite convenient to have these logs come out with the STEP log as well.
    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EXPRESSIONS | LIBLLDB_LOG_STEP));

    if (m_jit_start_addr != LLDB_INVALID_ADDRESS || m_can_interpret)
    {
        lldb::addr_t struct_address = LLDB_INVALID_ADDRESS;

        if (!PrepareToExecuteJITExpression (error_stream, exe_ctx, struct_address))
        {
            error_stream.Printf("Errored out in %s, couldn't PrepareToExecuteJITExpression", __FUNCTION__);
            return lldb::eExpressionSetupError;
        }

        lldb::addr_t function_stack_bottom = LLDB_INVALID_ADDRESS;
        lldb::addr_t function_stack_top = LLDB_INVALID_ADDRESS;

        if (m_can_interpret)
        {
            llvm::Module *module = m_execution_unit_sp->GetModule();
            llvm::Function *function = m_execution_unit_sp->GetFunction();

            if (!module || !function)
            {
                error_stream.Printf("Supposed to interpret, but nothing is there");
                return lldb::eExpressionSetupError;
            }

            Error interpreter_error;

            std::vector<lldb::addr_t> args;
            
            if (!AddInitialArguments(exe_ctx, args, error_stream))
            {
                error_stream.Printf ("Errored out in %s, couldn't AddInitialArguments", __FUNCTION__);
                return lldb::eExpressionSetupError;
            }
            
            args.push_back(struct_address);

            function_stack_bottom = m_stack_frame_bottom;
            function_stack_top = m_stack_frame_top;

            IRInterpreter::Interpret (*module,
                                      *function,
                                      args,
                                      *m_execution_unit_sp.get(),
                                      interpreter_error,
                                      function_stack_bottom,
                                      function_stack_top,
                                      exe_ctx);

            if (!interpreter_error.Success())
            {
                error_stream.Printf("Supposed to interpret, but failed: %s", interpreter_error.AsCString());
                return lldb::eExpressionDiscarded;
            }
        }
        else
        {
            if (!exe_ctx.HasThreadScope())
            {
                error_stream.Printf("UserExpression::Execute called with no thread selected.");
                return lldb::eExpressionSetupError;
            }

            Address wrapper_address (m_jit_start_addr);

            std::vector<lldb::addr_t> args;
            
            if (!AddInitialArguments(exe_ctx, args, error_stream))
            {
                error_stream.Printf ("Errored out in %s, couldn't AddInitialArguments", __FUNCTION__);
                return lldb::eExpressionSetupError;
            }

            args.push_back(struct_address);
         
            lldb::ThreadPlanSP call_plan_sp(new ThreadPlanCallUserExpression (exe_ctx.GetThreadRef(),
                                                                              wrapper_address,
                                                                              args,
                                                                              options,
                                                                              shared_ptr_to_me));

            if (!call_plan_sp || !call_plan_sp->ValidatePlan (&error_stream))
                return lldb::eExpressionSetupError;

            ThreadPlanCallUserExpression *user_expression_plan = static_cast<ThreadPlanCallUserExpression *>(call_plan_sp.get());

            lldb::addr_t function_stack_pointer = user_expression_plan->GetFunctionStackPointer();

            function_stack_bottom = function_stack_pointer - HostInfo::GetPageSize();
            function_stack_top = function_stack_pointer;

            if (log)
                log->Printf("-- [UserExpression::Execute] Execution of expression begins --");

            if (exe_ctx.GetProcessPtr())
                exe_ctx.GetProcessPtr()->SetRunningUserExpression(true);

            lldb::ExpressionResults execution_result = exe_ctx.GetProcessRef().RunThreadPlan (exe_ctx,
                                                                                       call_plan_sp,
                                                                                       options,
                                                                                       error_stream);

            if (exe_ctx.GetProcessPtr())
                exe_ctx.GetProcessPtr()->SetRunningUserExpression(false);

            if (log)
                log->Printf("-- [UserExpression::Execute] Execution of expression completed --");

            if (execution_result == lldb::eExpressionInterrupted || execution_result == lldb::eExpressionHitBreakpoint)
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
                    error_stream.PutCString ("Execution was interrupted.");

                if ((execution_result == lldb::eExpressionInterrupted && options.DoesUnwindOnError())
                    || (execution_result == lldb::eExpressionHitBreakpoint && options.DoesIgnoreBreakpoints()))
                    error_stream.PutCString ("\nThe process has been returned to the state before expression evaluation.");
                else
                {
                    if (execution_result == lldb::eExpressionHitBreakpoint)
                        user_expression_plan->TransferExpressionOwnership();
                    error_stream.PutCString ("\nThe process has been left at the point where it was interrupted, "
                                             "use \"thread return -x\" to return to the state before expression evaluation.");
                }

                return execution_result;
            }
            else if (execution_result == lldb::eExpressionStoppedForDebug)
            {
                    error_stream.PutCString ("Execution was halted at the first instruction of the expression "
                                             "function because \"debug\" was requested.\n"
                                             "Use \"thread return -x\" to return to the state before expression evaluation.");
                    return execution_result;
            }
            else if (execution_result != lldb::eExpressionCompleted)
            {
                error_stream.Printf ("Couldn't execute function; result was %s\n", Process::ExecutionResultAsCString (execution_result));
                return execution_result;
            }
        }

        if  (FinalizeJITExecution (error_stream, exe_ctx, result, function_stack_bottom, function_stack_top))
        {
            return lldb::eExpressionCompleted;
        }
        else
        {
            return lldb::eExpressionResultUnavailable;
        }
    }
    else
    {
        error_stream.Printf("Expression can't be run, because there is no JIT compiled function");
        return lldb::eExpressionSetupError;
    }
}

lldb::ExpressionResults
UserExpression::Evaluate (ExecutionContext &exe_ctx,
                               const EvaluateExpressionOptions& options,
                               const char *expr_cstr,
                               const char *expr_prefix,
                               lldb::ValueObjectSP &result_valobj_sp,
                               Error &error)
{
    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EXPRESSIONS | LIBLLDB_LOG_STEP));

    lldb_private::ExecutionPolicy execution_policy = options.GetExecutionPolicy();
    const lldb::LanguageType language = options.GetLanguage();
    const ResultType desired_type = options.DoesCoerceToId() ? UserExpression::eResultTypeId : UserExpression::eResultTypeAny;
    lldb::ExpressionResults execution_results = lldb::eExpressionSetupError;
    
    Target *target = exe_ctx.GetTargetPtr();
    if (!target)
    {
        if (log)
            log->Printf("== [UserExpression::Evaluate] Passed a NULL target, can't run expressions.");
        return lldb::eExpressionSetupError;
    }

    Process *process = exe_ctx.GetProcessPtr();

    if (process == NULL || process->GetState() != lldb::eStateStopped)
    {
        if (execution_policy == eExecutionPolicyAlways)
        {
            if (log)
                log->Printf("== [UserExpression::Evaluate] Expression may not run, but is not constant ==");

            error.SetErrorString ("expression needed to run but couldn't");

            return execution_results;
        }
    }

    if (process == NULL || !process->CanJIT())
        execution_policy = eExecutionPolicyNever;

    const char *full_prefix = NULL;
    const char *option_prefix = options.GetPrefix();
    std::string full_prefix_storage;
    if (expr_prefix && option_prefix)
    {
        full_prefix_storage.assign(expr_prefix);
        full_prefix_storage.append(option_prefix);
        if (!full_prefix_storage.empty())
            full_prefix = full_prefix_storage.c_str();
    }
    else if (expr_prefix)
        full_prefix = expr_prefix;
    else
        full_prefix = option_prefix;

    lldb::UserExpressionSP user_expression_sp(target->GetUserExpressionForLanguage (expr_cstr,
                                                                                    full_prefix,
                                                                                    language,
                                                                                    desired_type,
                                                                                    error));
    if (error.Fail())
    {
        if (log)
            log->Printf ("== [UserExpression::Evaluate] Getting expression: %s ==", error.AsCString());
        return lldb::eExpressionSetupError;
    }
 
    StreamString error_stream;

    if (log)
        log->Printf("== [UserExpression::Evaluate] Parsing expression %s ==", expr_cstr);

    const bool keep_expression_in_memory = true;
    const bool generate_debug_info = options.GetGenerateDebugInfo();

    if (options.InvokeCancelCallback (lldb::eExpressionEvaluationParse))
    {
        error.SetErrorString ("expression interrupted by callback before parse");
        result_valobj_sp = ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(), error);
        return lldb::eExpressionInterrupted;
    }

    if (!user_expression_sp->Parse (error_stream,
                                    exe_ctx,
                                    execution_policy,
                                    keep_expression_in_memory,
                                    generate_debug_info))
    {
        execution_results = lldb::eExpressionParseError;
        if (error_stream.GetString().empty())
            error.SetExpressionError (execution_results, "expression failed to parse, unknown error");
        else
            error.SetExpressionError (execution_results, error_stream.GetString().c_str());
    }
    else
    {
        lldb::ExpressionVariableSP expr_result;

        if (execution_policy == eExecutionPolicyNever &&
            !user_expression_sp->CanInterpret())
        {
            if (log)
                log->Printf("== [UserExpression::Evaluate] Expression may not run, but is not constant ==");

            if (error_stream.GetString().empty())
                error.SetExpressionError (lldb::eExpressionSetupError, "expression needed to run but couldn't");
        }
        else
        {
            if (options.InvokeCancelCallback (lldb::eExpressionEvaluationExecution))
            {
                error.SetExpressionError (lldb::eExpressionInterrupted, "expression interrupted by callback before execution");
                result_valobj_sp = ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(), error);
                return lldb::eExpressionInterrupted;
            }

            error_stream.GetString().clear();

            if (log)
                log->Printf("== [UserExpression::Evaluate] Executing expression ==");

            execution_results = user_expression_sp->Execute (error_stream,
                                                             exe_ctx,
                                                             options,
                                                             user_expression_sp,
                                                             expr_result);

            if (options.GetResultIsInternal() && expr_result && process)
            {
                process->GetTarget().GetPersistentVariables().RemovePersistentVariable (expr_result);
            }

            if (execution_results != lldb::eExpressionCompleted)
            {
                if (log)
                    log->Printf("== [UserExpression::Evaluate] Execution completed abnormally ==");

                if (error_stream.GetString().empty())
                    error.SetExpressionError (execution_results, "expression failed to execute, unknown error");
                else
                    error.SetExpressionError (execution_results, error_stream.GetString().c_str());
            }
            else
            {
                if (expr_result)
                {
                    result_valobj_sp = expr_result->GetValueObject();

                    if (log)
                        log->Printf("== [UserExpression::Evaluate] Execution completed normally with result %s ==",
                                    result_valobj_sp->GetValueAsCString());
                }
                else
                {
                    if (log)
                        log->Printf("== [UserExpression::Evaluate] Execution completed normally with no result ==");

                    error.SetError(UserExpression::kNoResult, lldb::eErrorTypeGeneric);
                }
            }
        }
    }

    if (options.InvokeCancelCallback(lldb::eExpressionEvaluationComplete))
    {
        error.SetExpressionError (lldb::eExpressionInterrupted, "expression interrupted by callback after complete");
        return lldb::eExpressionInterrupted;
    }

    if (result_valobj_sp.get() == NULL)
    {
        result_valobj_sp = ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(), error);
    }

    return execution_results;
}
