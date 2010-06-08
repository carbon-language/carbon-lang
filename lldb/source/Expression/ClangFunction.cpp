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
#include "clang/Frontend/CodeGenAction.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/Module.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "llvm/ADT/StringRef.h"

// Project includes
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallFunction.h"
#include "lldb/Core/Log.h"

using namespace lldb_private;
//----------------------------------------------------------------------
// ClangFunction constructor
//----------------------------------------------------------------------
ClangFunction::ClangFunction(const char *target_triple, ClangASTContext *ast_context, void *return_qualtype, const Address& functionAddress, const ValueList &arg_value_list) :
    ClangExpression (target_triple, NULL),
    m_function_addr (functionAddress),
    m_function_ptr (NULL),
    m_arg_values (arg_value_list),
    m_clang_ast_context (ast_context),
    m_function_return_qual_type(return_qualtype),
    m_wrapper_function_name ("__lldb_caller_function"),
    m_wrapper_struct_name ("__lldb_caller_struct"),
    m_return_offset(0),
    m_compiled (false),
    m_JITted (false)
{
}

ClangFunction::ClangFunction(const char *target_triple, Function &function, ClangASTContext *ast_context, const ValueList &arg_value_list) :
    ClangExpression (target_triple, NULL),
    m_function_ptr (&function),
    m_arg_values (arg_value_list),
    m_clang_ast_context (ast_context),
    m_function_return_qual_type (NULL),
    m_wrapper_function_name ("__lldb_function_caller"),
    m_wrapper_struct_name ("__lldb_caller_struct"),
    m_return_offset(0),
    m_compiled (false),
    m_JITted (false)
{
    m_function_addr = m_function_ptr->GetAddressRange().GetBaseAddress();
    m_function_return_qual_type = m_function_ptr->GetReturnType().GetOpaqueClangQualType();
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
    // FIXME: How does clang tell us there's no return value?  We need to handle that case.
    unsigned num_errors = 0;
    
    if (!m_compiled)
    {
        std::string return_type_str = ClangASTContext::GetTypeName(m_function_return_qual_type);
        
        // Cons up the function we're going to wrap our call in, then compile it...
        // We declare the function "extern "C"" because the compiler might be in C++
        // mode which would mangle the name and then we couldn't find it again...
        std::string expression;
        expression.append ("extern \"C\" void ");
        expression.append (m_wrapper_function_name);
        expression.append (" (void *input)\n{\n    struct ");
        expression.append (m_wrapper_struct_name);
        expression.append (" \n  {\n");
        expression.append ("    ");
        expression.append (return_type_str);
        expression.append (" (*fn_ptr) (");

        // Get the number of arguments.  If we have a function type and it is prototyped,
        // trust that, otherwise use the values we were given.

        // FIXME: This will need to be extended to handle Variadic functions.  We'll need
        // to pull the defined arguments out of the function, then add the types from the
        // arguments list for the variable arguments.

        size_t num_args = -1;
        bool trust_function = false;
        // GetArgumentCount returns -1 for an unprototyped function.
        if (m_function_ptr)
        {
            num_args = m_function_ptr->GetArgumentCount();
            if (num_args != -1)
                trust_function = true;
        }

        if (num_args == -1)
            num_args = m_arg_values.GetSize();

        std::string args_buffer;  // This one stores the definition of all the args in "struct caller".
        std::string args_list_buffer;  // This one stores the argument list called from the structure.
        for (int i = 0; i < num_args; i++)
        {
            const char *type_string;
            std::string type_stdstr;

            if (trust_function)
            {
                type_string = m_function_ptr->GetArgumentTypeAtIndex(i).GetName().AsCString();
            }
            else
            {
                Value *arg_value = m_arg_values.GetValueAtIndex(i);
                void *clang_qual_type = arg_value->GetOpaqueClangQualType ();
                if (clang_qual_type != NULL)
                {
                    type_stdstr = ClangASTContext::GetTypeName(clang_qual_type);
                    type_string = type_stdstr.c_str();
                }
                else
                {   
                    errors.Printf("Could not determine type of input value %d.", i);
                    return 1;
                }
            }


            expression.append (type_string);
            if (i < num_args - 1)
                expression.append (", ");

            char arg_buf[32];
            args_buffer.append ("    ");
            args_buffer.append (type_string);
            snprintf(arg_buf, 31, "arg_%d", i);
            args_buffer.push_back (' ');
            args_buffer.append (arg_buf);
            args_buffer.append (";\n");

            args_list_buffer.append ("__lldb_fn_data->");
            args_list_buffer.append (arg_buf);
            if (i < num_args - 1)
                args_list_buffer.append (", ");

        }
        expression.append (");\n"); // Close off the function calling prototype.

        expression.append (args_buffer);

        expression.append ("    ");
        expression.append (return_type_str);
        expression.append (" return_value;");
        expression.append ("\n  };\n  struct ");
        expression.append (m_wrapper_struct_name);
        expression.append ("* __lldb_fn_data = (struct ");
        expression.append (m_wrapper_struct_name);
        expression.append (" *) input;\n");

        expression.append ("  __lldb_fn_data->return_value = __lldb_fn_data->fn_ptr (");
        expression.append (args_list_buffer);
        expression.append (");\n}\n");

        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
        if (log)
            log->Printf ("Expression: \n\n%s\n\n", expression.c_str());
            
        // Okay, now compile this expression:
        num_errors = ParseBareExpression (expression.c_str(), errors);
        m_compiled = (num_errors == 0);
        
        if (m_compiled)
        {
            using namespace clang;
            CompilerInstance *compiler_instance = GetCompilerInstance();
            ASTContext &ast_context = compiler_instance->getASTContext();

            DeclarationName wrapper_func_name(&ast_context.Idents.get(m_wrapper_function_name.c_str()));
            FunctionDecl::lookup_result func_lookup = ast_context.getTranslationUnitDecl()->lookup(wrapper_func_name);
            if (func_lookup.first == func_lookup.second)
                return false;

            FunctionDecl *wrapper_func = dyn_cast<FunctionDecl> (*(func_lookup.first));
            if (!wrapper_func)
                return false;

            DeclarationName wrapper_struct_name(&ast_context.Idents.get(m_wrapper_struct_name.c_str()));
            RecordDecl::lookup_result struct_lookup = wrapper_func->lookup(wrapper_struct_name);
            if (struct_lookup.first == struct_lookup.second)
                return false;

            RecordDecl *wrapper_struct = dyn_cast<RecordDecl>(*(struct_lookup.first));

            if (!wrapper_struct)
                return false;

            m_struct_layout = &ast_context.getASTRecordLayout (wrapper_struct);
            if (!m_struct_layout)
            {
                m_compiled = false;
                return 1;
            }
            m_return_offset = m_struct_layout->getFieldOffset(m_struct_layout->getFieldCount() - 1);
            m_return_size = (m_struct_layout->getDataSize() - m_return_offset)/8;
        }
    }

    return num_errors;
}

bool
ClangFunction::WriteFunctionWrapper (ExecutionContext &exc_context, Stream &errors)
{
    Process *process = exc_context.process;

    if (process == NULL)
        return false;

    if (!m_JITted)
    {
        // Next we should JIT it and insert the result into the target program.
        if (!JITFunction (exc_context, m_wrapper_function_name.c_str()))
            return false;

        if (!WriteJITCode (exc_context))
            return false;

        m_JITted = true;
    }

    // Next get the call address for the function:
    m_wrapper_fun_addr = GetFunctionAddress (m_wrapper_function_name.c_str());
    if (m_wrapper_fun_addr == LLDB_INVALID_ADDRESS)
        return false;

    return true;
}

bool
ClangFunction::WriteFunctionArguments (ExecutionContext &exc_context, lldb::addr_t &args_addr_ref, Stream &errors)
{
    return WriteFunctionArguments(exc_context, args_addr_ref, m_function_addr, m_arg_values, errors);    
}

// FIXME: Assure that the ValueList we were passed in is consistent with the one that defined this function.

bool
ClangFunction::WriteFunctionArguments (ExecutionContext &exc_context, lldb::addr_t &args_addr_ref, Address function_address, ValueList &arg_values, Stream &errors)
{
    // Otherwise, allocate space for the argument passing struct, and write it.
    // We use the information in the expression parser AST to
    // figure out how to do this...
    // We should probably transcode this in this object so we can ditch the compiler instance
    // and all its associated data, and just keep the JITTed bytes.

    Error error;
    using namespace clang;
    ExecutionResults return_value = eExecutionSetupError;

    Process *process = exc_context.process;

    if (process == NULL)
        return return_value;
        
    uint64_t struct_size = m_struct_layout->getSize()/8; // Clang returns sizes in bytes.
    
    if (args_addr_ref == LLDB_INVALID_ADDRESS)
    {
        args_addr_ref = process->AllocateMemory(struct_size, lldb::ePermissionsReadable|lldb::ePermissionsWritable, error);
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

    // FIXME: This is fake, and just assumes that it matches that architecture.
    // Make a data extractor and put the address into the right byte order & size.

    uint64_t fun_addr = function_address.GetLoadAddress(exc_context.process);
    int first_offset = m_struct_layout->getFieldOffset(0)/8;
    process->WriteMemory(args_addr_ref + first_offset, &fun_addr, 8, error);

    // FIXME: We will need to extend this for Variadic functions.

    Error value_error;
    
    size_t num_args = arg_values.GetSize();
    if (num_args != m_arg_values.GetSize())
    {
        errors.Printf ("Wrong number of arguments - was: %d should be: %d", num_args, m_arg_values.GetSize());
        return false;
    }
    
    for (int i = 0; i < num_args; i++)
    {
        // FIXME: We should sanity check sizes.

        int offset = m_struct_layout->getFieldOffset(i+1)/8; // Clang sizes are in bytes.
        Value *arg_value = arg_values.GetValueAtIndex(i);
        
        // FIXME: For now just do scalars:
        
        // Special case: if it's a pointer, don't do anything (the ABI supports passing cstrings)
        
        if (arg_value->GetValueType() == Value::eValueTypeHostAddress &&
            arg_value->GetContextType() == Value::eContextTypeOpaqueClangQualType &&
            ClangASTContext::IsPointerType(arg_value->GetValueOpaqueClangQualType()))
            continue;
        
        const Scalar &arg_scalar = arg_value->ResolveValue(&exc_context, m_clang_ast_context->getASTContext());

        int byte_size = arg_scalar.GetByteSize();
        std::vector<uint8_t> buffer;
        buffer.resize(byte_size);
        DataExtractor value_data;
        arg_scalar.GetData (value_data);
        value_data.ExtractBytes(0, byte_size, process->GetByteOrder(), buffer.data());
        process->WriteMemory(args_addr_ref + offset, buffer.data(), byte_size, error);
    }

    return true;
}

bool
ClangFunction::InsertFunction (ExecutionContext &exc_context, lldb::addr_t &args_addr_ref, Stream &errors)
{
    using namespace clang;
    
    if (CompileFunction(errors) != 0)
        return false;
    if (!WriteFunctionWrapper(exc_context, errors))
        return false;
    if (!WriteFunctionArguments(exc_context, args_addr_ref, errors))
        return false;

    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
    if (log)
        log->Printf ("Call Address: 0x%llx Struct Address: 0x%llx.\n", m_wrapper_fun_addr, args_addr_ref);
        
    return true;
}

ThreadPlan *
ClangFunction::GetThreadPlanToCallFunction (ExecutionContext &exc_context, lldb::addr_t &args_addr, Stream &errors, bool stop_others, bool discard_on_error)
{
    // FIXME: Use the errors Stream for better error reporting.

    Process *process = exc_context.process;

    if (process == NULL)
    {
        errors.Printf("Can't call a function without a process.");
        return NULL;
    }

    // Okay, now run the function:

    Address wrapper_address (NULL, m_wrapper_fun_addr);
    ThreadPlan *new_plan = new ThreadPlanCallFunction (*exc_context.thread, 
                                          wrapper_address,
                                          args_addr,
                                          stop_others, discard_on_error);
    return new_plan;
}

bool
ClangFunction::FetchFunctionResults (ExecutionContext &exc_context, lldb::addr_t args_addr, Value &ret_value)
{
    // Read the return value - it is the last field in the struct:
    // FIXME: How does clang tell us there's no return value?  We need to handle that case.
    
    std::vector<uint8_t> data_buffer;
    data_buffer.resize(m_return_size);
    Process *process = exc_context.process;
    Error error;
    size_t bytes_read = process->ReadMemory(args_addr + m_return_offset/8, data_buffer.data(), m_return_size, error);

    if (bytes_read == 0)
    {
        return false;
    }

    if (bytes_read < m_return_size)
        return false;

    DataExtractor data(data_buffer.data(), m_return_size, process->GetByteOrder(), process->GetAddressByteSize());
    // FIXME: Assuming an integer scalar for now:
    
    uint32_t offset = 0;
    uint64_t return_integer = data.GetMaxU64(&offset, m_return_size);
    
    ret_value.SetContext (Value::eContextTypeOpaqueClangQualType, m_function_return_qual_type);
    ret_value.SetValueType(Value::eValueTypeScalar);
    ret_value.GetScalar() = return_integer;
    return true;
}

void
ClangFunction::DeallocateFunctionResults (ExecutionContext &exc_context, lldb::addr_t args_addr)
{
    std::list<lldb::addr_t>::iterator pos;
    pos = std::find(m_wrapper_args_addrs.begin(), m_wrapper_args_addrs.end(), args_addr);
    if (pos != m_wrapper_args_addrs.end())
        m_wrapper_args_addrs.erase(pos);
    
    exc_context.process->DeallocateMemory(args_addr);
}

ClangFunction::ExecutionResults
ClangFunction::ExecuteFunction(ExecutionContext &exc_context, Stream &errors, Value &results)
{
    return ExecuteFunction (exc_context, errors, 1000, true, results);
}

ClangFunction::ExecutionResults
ClangFunction::ExecuteFunction(ExecutionContext &exc_context, Stream &errors, bool stop_others, Value &results)
{
    return ExecuteFunction (exc_context, NULL, errors, stop_others, NULL, false, results);
}

ClangFunction::ExecutionResults
ClangFunction::ExecuteFunction(
        ExecutionContext &exc_context, 
        Stream &errors, 
        uint32_t single_thread_timeout_usec, 
        bool try_all_threads, 
        Value &results)
{
    return ExecuteFunction (exc_context, NULL, errors, true, single_thread_timeout_usec, try_all_threads, results);
}

ClangFunction::ExecutionResults
ClangFunction::ExecuteFunction(
        ExecutionContext &exc_context, 
        lldb::addr_t *args_addr_ptr, 
        Stream &errors, 
        bool stop_others, 
        uint32_t single_thread_timeout_usec, 
        bool try_all_threads, 
        Value &results)
{
    using namespace clang;
    ExecutionResults return_value = eExecutionSetupError;
    Process *process = exc_context.process;
    
    lldb::addr_t args_addr;
    
    if (args_addr_ptr != NULL)
        args_addr = *args_addr_ptr;
    else
        args_addr = LLDB_INVALID_ADDRESS;
        
    if (CompileFunction(errors) != 0)
        return eExecutionSetupError;
    
    if (args_addr == LLDB_INVALID_ADDRESS)
    {
        if (!InsertFunction(exc_context, args_addr, errors))
            return eExecutionSetupError;
    }
    
    
    lldb::ThreadPlanSP call_plan_sp(GetThreadPlanToCallFunction(exc_context, args_addr, errors, stop_others, false));
    
    ThreadPlanCallFunction *call_plan_ptr = static_cast<ThreadPlanCallFunction *> (call_plan_sp.get());
    
    if (args_addr_ptr != NULL)
        *args_addr_ptr = args_addr;
    
    if (call_plan_sp == NULL)
        return return_value;

    call_plan_sp->SetPrivate(true);
    exc_context.thread->QueueThreadPlan(call_plan_sp, true);

    // We need to call the function synchronously, so spin waiting for it to return.
    // If we get interrupted while executing, we're going to lose our context, and
    // won't be able to gather the result at this point.

    TimeValue* timeout_ptr = NULL;
    TimeValue real_timeout;
    if (single_thread_timeout_usec != 0)
    {
        real_timeout = TimeValue::Now();
        real_timeout.OffsetWithMicroSeconds(single_thread_timeout_usec);
        timeout_ptr = &real_timeout;
    }
    process->Resume ();

    
    while (1)
    {
        lldb::EventSP event_sp;

        // Now wait for the process to stop again:
        // FIXME: Probably want a time out.
        lldb::StateType stop_state =  process->WaitForStateChangedEvents (timeout_ptr, event_sp);
        if (stop_state == lldb::eStateInvalid && timeout_ptr != NULL)
        {
            // Right now this is the only way to tell we've timed out...
            // We should interrupt the process here...
            // Not really sure what to do if Halt fails here...
            Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
            if (log)
                log->Printf ("Running function with timeout: %d timed out, trying with all threads enabled.", single_thread_timeout_usec);

            if (process->Halt().Success())
            {
                timeout_ptr = NULL;
                
                lldb::StateType stop_state =  process->WaitForStateChangedEvents (timeout_ptr, event_sp);
                if (stop_state == lldb::eStateInvalid)
                {
                    errors.Printf ("Got an invalid stop state after halt.");
                }
                else if (stop_state != lldb::eStateStopped)
                {
                    StreamString s;
                    event_sp->Dump (&s);
                    
                    errors.Printf("Didn't get a stopped event after Halting the target, got: \"%s\"", s.GetData());
                }
                
                if (try_all_threads)
                {
                    // Between the time that we got the timeout and the time we halted, but target
                    // might have actually completed the plan.  If so, we're done.
                    if (exc_context.thread->IsThreadPlanDone (call_plan_sp.get()))
                    {
                        return_value = eExecutionCompleted;
                        break;
                    }
                

                    call_plan_ptr->SetStopOthers (false);
                    process->Resume();
                    continue;
                }
                else
                    return eExecutionInterrupted;
            }
        }
        if (stop_state == lldb::eStateRunning || stop_state == lldb::eStateStepping)
            continue;

        if (exc_context.thread->IsThreadPlanDone (call_plan_sp.get()))
        {
            return_value = eExecutionCompleted;
            break;
        }
        else if (exc_context.thread->WasThreadPlanDiscarded (call_plan_sp.get()))
        {
            return_value = eExecutionDiscarded;
            break;
        }
        else
        {
            return_value = eExecutionInterrupted;
            break;
        }

    }

    if (return_value != eExecutionCompleted)
        return return_value;

    FetchFunctionResults(exc_context, args_addr, results);
    
    if (args_addr_ptr == NULL)
        DeallocateFunctionResults(exc_context, args_addr);
        
    return eExecutionCompleted;
}

ClangFunction::ExecutionResults
ClangFunction::ExecuteFunctionWithABI(ExecutionContext &exc_context, Stream &errors, Value &results)
{
    // FIXME: Use the errors Stream for better error reporting. 
    using namespace clang;
    ExecutionResults return_value = eExecutionSetupError;
    
    Process *process = exc_context.process;
    
    if (process == NULL)
    {
        errors.Printf("Can't call a function without a process.");
        return return_value;
    }
    
    //unsigned int num_args = m_arg_values.GetSize();
    //unsigned int arg_index;
    
    //for (arg_index = 0; arg_index < num_args; ++arg_index)
    //    m_arg_values.GetValueAtIndex(arg_index)->ResolveValue(&exc_context, GetASTContext());
    
    ThreadPlan *call_plan = exc_context.thread->QueueThreadPlanForCallFunction (false,
                                                                                m_function_addr,
                                                                                m_arg_values,
                                                                                true);
    if (call_plan == NULL)
        return return_value;
    
    call_plan->SetPrivate(true);
    
    // We need to call the function synchronously, so spin waiting for it to return.
    // If we get interrupted while executing, we're going to lose our context, and 
    // won't be able to gather the result at this point.
    
    process->Resume ();
    
    while (1)
    {
        lldb::EventSP event_sp;
        
        // Now wait for the process to stop again:
        // FIXME: Probably want a time out.
        lldb::StateType stop_state =  process->WaitForStateChangedEvents (NULL, event_sp);
        if (stop_state == lldb::eStateRunning || stop_state == lldb::eStateStepping)
            continue;
        
        if (exc_context.thread->IsThreadPlanDone (call_plan))
        {
            return_value = eExecutionCompleted;
            break;
        }
        else if (exc_context.thread->WasThreadPlanDiscarded (call_plan))
        {
            return_value = eExecutionDiscarded;
            break;
        }
        else
        {
            return_value = eExecutionInterrupted;
            break;
        }
        
    }
    
    return eExecutionCompleted;
}
