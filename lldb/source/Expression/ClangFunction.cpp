//===-- ClangFunction.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// C Includes
// C++ Includes
// Other libraries and framework includes
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/Module.h"

// Project includes
#include "lldb/Expression/ASTStructExtractor.h"
#include "lldb/Expression/ClangExpressionParser.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/State.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallFunction.h"
#include "lldb/Core/Log.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// ClangFunction constructor
//----------------------------------------------------------------------
ClangFunction::ClangFunction 
(
    ExecutionContextScope &exe_scope,
    const ClangASTType &return_type, 
    const Address& functionAddress, 
    const ValueList &arg_value_list
) :
    m_function_ptr (NULL),
    m_function_addr (functionAddress),
    m_function_return_type(return_type),
    m_wrapper_function_name ("__lldb_caller_function"),
    m_wrapper_struct_name ("__lldb_caller_struct"),
    m_wrapper_args_addrs (),
    m_arg_values (arg_value_list),
    m_compiled (false),
    m_JITted (false)
{
    m_jit_process_wp = lldb::ProcessWP(exe_scope.CalculateProcess());
    // Can't make a ClangFunction without a process.
    assert (m_jit_process_wp.lock());
}

ClangFunction::ClangFunction
(
    ExecutionContextScope &exe_scope,
    Function &function, 
    ClangASTContext *ast_context, 
    const ValueList &arg_value_list
) :
    m_function_ptr (&function),
    m_function_addr (),
    m_function_return_type (),
    m_wrapper_function_name ("__lldb_function_caller"),
    m_wrapper_struct_name ("__lldb_caller_struct"),
    m_wrapper_args_addrs (),
    m_arg_values (arg_value_list),
    m_compiled (false),
    m_JITted (false)
{
    m_jit_process_wp = lldb::ProcessWP(exe_scope.CalculateProcess());
    // Can't make a ClangFunction without a process.
    assert (m_jit_process_wp.lock());

    m_function_addr = m_function_ptr->GetAddressRange().GetBaseAddress();
    m_function_return_type = m_function_ptr->GetClangType().GetFunctionReturnType();
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ClangFunction::~ClangFunction()
{
}

unsigned
ClangFunction::CompileFunction (Stream &errors)
{
    if (m_compiled)
        return 0;
    
    // FIXME: How does clang tell us there's no return value?  We need to handle that case.
    unsigned num_errors = 0;
    
    std::string return_type_str (m_function_return_type.GetTypeName().AsCString(""));
    
    // Cons up the function we're going to wrap our call in, then compile it...
    // We declare the function "extern "C"" because the compiler might be in C++
    // mode which would mangle the name and then we couldn't find it again...
    m_wrapper_function_text.clear();
    m_wrapper_function_text.append ("extern \"C\" void ");
    m_wrapper_function_text.append (m_wrapper_function_name);
    m_wrapper_function_text.append (" (void *input)\n{\n    struct ");
    m_wrapper_function_text.append (m_wrapper_struct_name);
    m_wrapper_function_text.append (" \n  {\n");
    m_wrapper_function_text.append ("    ");
    m_wrapper_function_text.append (return_type_str);
    m_wrapper_function_text.append (" (*fn_ptr) (");

    // Get the number of arguments.  If we have a function type and it is prototyped,
    // trust that, otherwise use the values we were given.

    // FIXME: This will need to be extended to handle Variadic functions.  We'll need
    // to pull the defined arguments out of the function, then add the types from the
    // arguments list for the variable arguments.

    uint32_t num_args = UINT32_MAX;
    bool trust_function = false;
    // GetArgumentCount returns -1 for an unprototyped function.
    ClangASTType function_clang_type;
    if (m_function_ptr)
    {
        function_clang_type = m_function_ptr->GetClangType();
        if (function_clang_type)
        {
            int num_func_args = function_clang_type.GetFunctionArgumentCount();
            if (num_func_args >= 0)
            {
                trust_function = true;
                num_args = num_func_args;
            }
        }
    }

    if (num_args == UINT32_MAX)
        num_args = m_arg_values.GetSize();

    std::string args_buffer;  // This one stores the definition of all the args in "struct caller".
    std::string args_list_buffer;  // This one stores the argument list called from the structure.
    for (size_t i = 0; i < num_args; i++)
    {
        std::string type_name;

        if (trust_function)
        {
            type_name = function_clang_type.GetFunctionArgumentTypeAtIndex(i).GetTypeName().AsCString("");
        }
        else
        {
            ClangASTType clang_qual_type = m_arg_values.GetValueAtIndex(i)->GetClangType ();
            if (clang_qual_type)
            {
                type_name = clang_qual_type.GetTypeName().AsCString("");
            }
            else
            {   
                errors.Printf("Could not determine type of input value %" PRIu64 ".", (uint64_t)i);
                return 1;
            }
        }

        m_wrapper_function_text.append (type_name);
        if (i < num_args - 1)
            m_wrapper_function_text.append (", ");

        char arg_buf[32];
        args_buffer.append ("    ");
        args_buffer.append (type_name);
        snprintf(arg_buf, 31, "arg_%" PRIu64, (uint64_t)i);
        args_buffer.push_back (' ');
        args_buffer.append (arg_buf);
        args_buffer.append (";\n");

        args_list_buffer.append ("__lldb_fn_data->");
        args_list_buffer.append (arg_buf);
        if (i < num_args - 1)
            args_list_buffer.append (", ");

    }
    m_wrapper_function_text.append (");\n"); // Close off the function calling prototype.

    m_wrapper_function_text.append (args_buffer);

    m_wrapper_function_text.append ("    ");
    m_wrapper_function_text.append (return_type_str);
    m_wrapper_function_text.append (" return_value;");
    m_wrapper_function_text.append ("\n  };\n  struct ");
    m_wrapper_function_text.append (m_wrapper_struct_name);
    m_wrapper_function_text.append ("* __lldb_fn_data = (struct ");
    m_wrapper_function_text.append (m_wrapper_struct_name);
    m_wrapper_function_text.append (" *) input;\n");

    m_wrapper_function_text.append ("  __lldb_fn_data->return_value = __lldb_fn_data->fn_ptr (");
    m_wrapper_function_text.append (args_list_buffer);
    m_wrapper_function_text.append (");\n}\n");

    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    if (log)
        log->Printf ("Expression: \n\n%s\n\n", m_wrapper_function_text.c_str());
        
    // Okay, now compile this expression
    
    lldb::ProcessSP jit_process_sp(m_jit_process_wp.lock());
    if (jit_process_sp)
    {
        m_parser.reset(new ClangExpressionParser(jit_process_sp.get(), *this));
        
        num_errors = m_parser->Parse (errors);
    }
    else
    {
        errors.Printf("no process - unable to inject function");
        num_errors = 1;
    }
    
    m_compiled = (num_errors == 0);
    
    if (!m_compiled)
        return num_errors;

    return num_errors;
}

bool
ClangFunction::WriteFunctionWrapper (ExecutionContext &exe_ctx, Stream &errors)
{
    Process *process = exe_ctx.GetProcessPtr();

    if (!process)
        return false;
    
    lldb::ProcessSP jit_process_sp(m_jit_process_wp.lock());
    
    if (process != jit_process_sp.get())
        return false;
    
    if (!m_compiled)
        return false;

    if (m_JITted)
        return true;
        
    bool can_interpret = false; // should stay that way
    
    Error jit_error (m_parser->PrepareForExecution (m_jit_start_addr,
                                                    m_jit_end_addr,
                                                    m_execution_unit_ap,
                                                    exe_ctx, 
                                                    can_interpret,
                                                    eExecutionPolicyAlways));
    
    if (!jit_error.Success())
        return false;
    
    if (process && m_jit_start_addr)
        m_jit_process_wp = lldb::ProcessWP(process->shared_from_this());
    
    m_JITted = true;

    return true;
}

bool
ClangFunction::WriteFunctionArguments (ExecutionContext &exe_ctx, lldb::addr_t &args_addr_ref, Stream &errors)
{
    return WriteFunctionArguments(exe_ctx, args_addr_ref, m_function_addr, m_arg_values, errors);    
}

// FIXME: Assure that the ValueList we were passed in is consistent with the one that defined this function.

bool
ClangFunction::WriteFunctionArguments (ExecutionContext &exe_ctx, 
                                       lldb::addr_t &args_addr_ref, 
                                       Address function_address, 
                                       ValueList &arg_values, 
                                       Stream &errors)
{
    // All the information to reconstruct the struct is provided by the
    // StructExtractor.
    if (!m_struct_valid)
    {
        errors.Printf("Argument information was not correctly parsed, so the function cannot be called.");
        return false;
    }
        
    Error error;
    using namespace clang;
    ExecutionResults return_value = eExecutionSetupError;

    Process *process = exe_ctx.GetProcessPtr();

    if (process == NULL)
        return return_value;

    lldb::ProcessSP jit_process_sp(m_jit_process_wp.lock());
    
    if (process != jit_process_sp.get())
        return false;
                
    if (args_addr_ref == LLDB_INVALID_ADDRESS)
    {
        args_addr_ref = process->AllocateMemory(m_struct_size, lldb::ePermissionsReadable|lldb::ePermissionsWritable, error);
        if (args_addr_ref == LLDB_INVALID_ADDRESS)
            return false;
        m_wrapper_args_addrs.push_back (args_addr_ref);
    } 
    else 
    {
        // Make sure this is an address that we've already handed out.
        if (find (m_wrapper_args_addrs.begin(), m_wrapper_args_addrs.end(), args_addr_ref) == m_wrapper_args_addrs.end())
        {
            return false;
        }
    }

    // TODO: verify fun_addr needs to be a callable address
    Scalar fun_addr (function_address.GetCallableLoadAddress(exe_ctx.GetTargetPtr()));
    uint64_t first_offset = m_member_offsets[0];
    process->WriteScalarToMemory(args_addr_ref + first_offset, fun_addr, process->GetAddressByteSize(), error);

    // FIXME: We will need to extend this for Variadic functions.

    Error value_error;
    
    size_t num_args = arg_values.GetSize();
    if (num_args != m_arg_values.GetSize())
    {
        errors.Printf ("Wrong number of arguments - was: %" PRIu64 " should be: %" PRIu64 "", (uint64_t)num_args, (uint64_t)m_arg_values.GetSize());
        return false;
    }
    
    for (size_t i = 0; i < num_args; i++)
    {
        // FIXME: We should sanity check sizes.

        uint64_t offset = m_member_offsets[i+1]; // Clang sizes are in bytes.
        Value *arg_value = arg_values.GetValueAtIndex(i);
        
        // FIXME: For now just do scalars:
        
        // Special case: if it's a pointer, don't do anything (the ABI supports passing cstrings)
        
        if (arg_value->GetValueType() == Value::eValueTypeHostAddress &&
            arg_value->GetContextType() == Value::eContextTypeInvalid &&
            arg_value->GetClangType().IsPointerType())
            continue;
        
        const Scalar &arg_scalar = arg_value->ResolveValue(&exe_ctx);

        if (!process->WriteScalarToMemory(args_addr_ref + offset, arg_scalar, arg_scalar.GetByteSize(), error))
            return false;
    }

    return true;
}

bool
ClangFunction::InsertFunction (ExecutionContext &exe_ctx, lldb::addr_t &args_addr_ref, Stream &errors)
{
    using namespace clang;
    
    if (CompileFunction(errors) != 0)
        return false;
    if (!WriteFunctionWrapper(exe_ctx, errors))
        return false;
    if (!WriteFunctionArguments(exe_ctx, args_addr_ref, errors))
        return false;

    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
        log->Printf ("Call Address: 0x%" PRIx64 " Struct Address: 0x%" PRIx64 ".\n", m_jit_start_addr, args_addr_ref);
        
    return true;
}

ThreadPlan *
ClangFunction::GetThreadPlanToCallFunction (ExecutionContext &exe_ctx, 
                                            lldb::addr_t args_addr,
                                            const EvaluateExpressionOptions &options,
                                            Stream &errors)
{
    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EXPRESSIONS | LIBLLDB_LOG_STEP));
    
    if (log)
        log->Printf("-- [ClangFunction::GetThreadPlanToCallFunction] Creating thread plan to call function --");
    
    // FIXME: Use the errors Stream for better error reporting.
    Thread *thread = exe_ctx.GetThreadPtr();
    if (thread == NULL)
    {
        errors.Printf("Can't call a function without a valid thread.");
        return NULL;
    }

    // Okay, now run the function:

    Address wrapper_address (m_jit_start_addr);
    
    lldb::addr_t args = { args_addr };
    
    ThreadPlan *new_plan = new ThreadPlanCallFunction (*thread, 
                                                       wrapper_address,
                                                       ClangASTType(),
                                                       args,
                                                       options);
    new_plan->SetIsMasterPlan(true);
    new_plan->SetOkayToDiscard (false);
    return new_plan;
}

bool
ClangFunction::FetchFunctionResults (ExecutionContext &exe_ctx, lldb::addr_t args_addr, Value &ret_value)
{
    // Read the return value - it is the last field in the struct:
    // FIXME: How does clang tell us there's no return value?  We need to handle that case.
    // FIXME: Create our ThreadPlanCallFunction with the return ClangASTType, and then use GetReturnValueObject
    // to fetch the value.  That way we can fetch any values we need.
    
    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EXPRESSIONS | LIBLLDB_LOG_STEP));
    
    if (log)
        log->Printf("-- [ClangFunction::FetchFunctionResults] Fetching function results --");
    
    Process *process = exe_ctx.GetProcessPtr();
    
    if (process == NULL)
        return false;

    lldb::ProcessSP jit_process_sp(m_jit_process_wp.lock());
    
    if (process != jit_process_sp.get())
        return false;
                
    Error error;
    ret_value.GetScalar() = process->ReadUnsignedIntegerFromMemory (args_addr + m_return_offset, m_return_size, 0, error);

    if (error.Fail())
        return false;

    ret_value.SetClangType(m_function_return_type);
    ret_value.SetValueType(Value::eValueTypeScalar);
    return true;
}

void
ClangFunction::DeallocateFunctionResults (ExecutionContext &exe_ctx, lldb::addr_t args_addr)
{
    std::list<lldb::addr_t>::iterator pos;
    pos = std::find(m_wrapper_args_addrs.begin(), m_wrapper_args_addrs.end(), args_addr);
    if (pos != m_wrapper_args_addrs.end())
        m_wrapper_args_addrs.erase(pos);
    
    exe_ctx.GetProcessRef().DeallocateMemory(args_addr);
}

ExecutionResults
ClangFunction::ExecuteFunction(
        ExecutionContext &exe_ctx, 
        lldb::addr_t *args_addr_ptr,
        const EvaluateExpressionOptions &options,
        Stream &errors, 
        Value &results)
{
    using namespace clang;
    ExecutionResults return_value = eExecutionSetupError;
    
    // ClangFunction::ExecuteFunction execution is always just to get the result.  Do make sure we ignore
    // breakpoints, unwind on error, and don't try to debug it.
    EvaluateExpressionOptions real_options = options;
    real_options.SetDebug(false);
    real_options.SetUnwindOnError(true);
    real_options.SetIgnoreBreakpoints(true);
    
    lldb::addr_t args_addr;
    
    if (args_addr_ptr != NULL)
        args_addr = *args_addr_ptr;
    else
        args_addr = LLDB_INVALID_ADDRESS;
        
    if (CompileFunction(errors) != 0)
        return eExecutionSetupError;
    
    if (args_addr == LLDB_INVALID_ADDRESS)
    {
        if (!InsertFunction(exe_ctx, args_addr, errors))
            return eExecutionSetupError;
    }

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EXPRESSIONS | LIBLLDB_LOG_STEP));

    if (log)
        log->Printf("== [ClangFunction::ExecuteFunction] Executing function ==");
    
    lldb::ThreadPlanSP call_plan_sp (GetThreadPlanToCallFunction (exe_ctx,
                                                                  args_addr,
                                                                  real_options,
                                                                  errors));
    if (!call_plan_sp)
        return eExecutionSetupError;
        
    // <rdar://problem/12027563> we need to make sure we record the fact that we are running an expression here
    // otherwise this fact will fail to be recorded when fetching an Objective-C object description
    if (exe_ctx.GetProcessPtr())
        exe_ctx.GetProcessPtr()->SetRunningUserExpression(true);
    
    return_value = exe_ctx.GetProcessRef().RunThreadPlan (exe_ctx,
                                                          call_plan_sp,
                                                          real_options,
                                                          errors);
    
    if (log)
    {
        if (return_value != eExecutionCompleted)
        {
            log->Printf("== [ClangFunction::ExecuteFunction] Execution completed abnormally ==");
        }
        else
        {
            log->Printf("== [ClangFunction::ExecuteFunction] Execution completed normally ==");
        }
    }
    
    if (exe_ctx.GetProcessPtr())
        exe_ctx.GetProcessPtr()->SetRunningUserExpression(false);
    
    if (args_addr_ptr != NULL)
        *args_addr_ptr = args_addr;
    
    if (return_value != eExecutionCompleted)
        return return_value;

    FetchFunctionResults(exe_ctx, args_addr, results);
    
    if (args_addr_ptr == NULL)
        DeallocateFunctionResults(exe_ctx, args_addr);
        
    return eExecutionCompleted;
}

clang::ASTConsumer *
ClangFunction::ASTTransformer (clang::ASTConsumer *passthrough)
{
    m_struct_extractor.reset(new ASTStructExtractor(passthrough, m_wrapper_struct_name.c_str(), *this));
    
    return m_struct_extractor.get();
}
