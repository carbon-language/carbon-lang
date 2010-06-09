//===-- ClangFunction.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_ClangFunction_h_
#define lldb_ClangFunction_h_

// C Includes
// C++ Includes
#include <vector>
#include <list>
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectList.h"

#include "lldb/Expression/ClangExpression.h"

// Right now, this is just a toy.  It calls a set function, with fixed
// values.

namespace clang
{
    class ASTRecordLayout;
}

namespace lldb_private
{

class ClangFunction : private ClangExpression
{
public:

    enum ExecutionResults
    {
        eExecutionSetupError,
        eExecutionCompleted,
        eExecutionDiscarded,
        eExecutionInterrupted,
        eExecutionTimedOut
    };
        
	//------------------------------------------------------------------
	// Constructors and Destructors
	//------------------------------------------------------------------
    // Usage Note:
    
    // A given ClangFunction object can handle any function with a common signature.  It can also be used to
    // set up any number of concurrent functions calls once it has been constructed.
    // When you construct it you pass in a particular function, information sufficient to determine the function signature 
    // and value list.
    // The simplest use of the ClangFunction is to construct the function, then call ExecuteFunction (context, errors, results). The function
    // will be called using the initial arguments, and the results determined for you, and all cleanup done.
    //
    // However, if you need to use the function caller in Thread Plans, you need to call the function on the plan stack.
    // In that case, you call GetThreadPlanToCallFunction, args_addr will be the location of the args struct, and after you are
    // done running this thread plan you can recover the results using FetchFunctionResults passing in the same value.
    // You are required to call InsertFunction before calling GetThreadPlanToCallFunction.
    //
    // You can also reuse the struct if you want, by calling ExecuteFunction but passing in args_addr_ptr primed to this value.
    //
    // You can also reuse the ClangFunction for the same signature but different function or argument values by rewriting the
    // Functions arguments with WriteFunctionArguments, and then calling ExecuteFunction passing in the same args_addr.
    //
    // Note, any of the functions below that take arg_addr_ptr, or arg_addr_ref, can be passed a pointer set to LLDB_INVALID_ADDRESS and 
    // new structure will be allocated and its address returned in that variable.
    // Any of the functions below that take arg_addr_ptr can be passed NULL, and the argument space will be managed for you.
    
	ClangFunction(const char *target_triple, Function &function_ptr, ClangASTContext *ast_context, const ValueList &arg_value_list);
    // This constructor takes its return type as a Clang QualType opaque pointer, and the ast_context it comes from.
    // FIXME: We really should be able to easily make a Type from the qualtype, and then just pass that in.
	ClangFunction(const char *target_triple, ClangASTContext *ast_context, void *return_qualtype, const Address& functionAddress, const ValueList &arg_value_list);
	virtual ~ClangFunction();

    unsigned CompileFunction (Stream &errors);
    
    // args_addr is a pointer to the address the addr will be filled with.  If the value on 
    // input is LLDB_INVALID_ADDRESS then a new address will be allocated, and returned in args_addr.
    // If args_addr is a value already returned from a previous call to InsertFunction, then 
    // the args structure at that address is overwritten. 
    // If any other value is returned, then we return false, and do nothing.
    bool InsertFunction (ExecutionContext &context, lldb::addr_t &args_addr_ref, Stream &errors);

    bool WriteFunctionWrapper (ExecutionContext &exec_ctx, Stream &errors);
    
    // This variant writes down the original function address and values to args_addr.
    bool WriteFunctionArguments (ExecutionContext &exec_ctx, lldb::addr_t &args_addr_ref, Stream &errors);
    
    // This variant writes down function_address and arg_value.
    bool WriteFunctionArguments (ExecutionContext &exc_context, lldb::addr_t &args_addr_ref, Address function_address, ValueList &arg_values, Stream &errors);

    //------------------------------------------------------------------
    /// Run the function this ClangFunction was created with.
    ///
    /// This simple version will run the function stopping other threads
    /// for a fixed timeout period (1000 usec) and if it does not complete,
    /// we halt the process and try with all threads running.
    ///
    /// @param[in] context
    ///     The thread & process in which this function will run.
    ///
    /// @param[in] errors
    ///     Errors will be written here if there are any.
    ///
    /// @param[out] results
    ///     The result value will be put here after running the function.
    ///
    /// @return
    ///     Returns one of the ExecutionResults enum indicating function call status.
    //------------------------------------------------------------------
    ExecutionResults ExecuteFunction(ExecutionContext &context, Stream &errors, Value &results);
    
    //------------------------------------------------------------------
    /// Run the function this ClangFunction was created with.
    ///
    /// This simple version will run the function obeying the stop_others
    /// argument.  There is no timeout.
    ///
    /// @param[in] context
    ///     The thread & process in which this function will run.
    ///
    /// @param[in] errors
    ///     Errors will be written here if there are any.
    ///
    /// @param[in] stop_others
    ///     If \b true, run only this thread, if \b false let all threads run.
    ///
    /// @param[out] results
    ///     The result value will be put here after running the function.
    ///
    /// @return
    ///     Returns one of the ExecutionResults enum indicating function call status.
    //------------------------------------------------------------------
    ExecutionResults ExecuteFunction(ExecutionContext &exc_context, Stream &errors, bool stop_others, Value &results);
    
    //------------------------------------------------------------------
    /// Run the function this ClangFunction was created with.
    ///
    /// This simple version will run the function on one thread.  If \a single_thread_timeout_usec
    /// is not zero, we time out after that timeout.  If \a try_all_threads is true, then we will
    /// resume with all threads on, otherwise we halt the process, and eExecutionInterrupted will be returned.
    ///
    /// @param[in] context
    ///     The thread & process in which this function will run.
    ///
    /// @param[in] errors
    ///     Errors will be written here if there are any.
    ///
    /// @param[in] single_thread_timeout_usec
    ///     If \b true, run only this thread, if \b false let all threads run.
    ///
    /// @param[in] try_all_threads
    ///     If \b true, run only this thread, if \b false let all threads run.
    ///
    /// @param[out] results
    ///     The result value will be put here after running the function.
    ///
    /// @return
    ///     Returns one of the ExecutionResults enum indicating function call status.
    //------------------------------------------------------------------
    ExecutionResults ExecuteFunction(ExecutionContext &context, Stream &errors, uint32_t single_thread_timeout_usec, bool try_all_threads, Value &results);
    
    //------------------------------------------------------------------
    /// Run the function this ClangFunction was created with.
    ///
    /// This is the full version.
    ///
    /// @param[in] context
    ///     The thread & process in which this function will run.
    ///
    /// @param[in] args_addr_ptr
    ///     If NULL, the function will take care of allocating & deallocating the wrapper
    ///     args structure.  Otherwise, if set to LLDB_INVALID_ADDRESS, a new structure
    ///     will be allocated, filled and the address returned to you.  You are responsible
    ///     for deallocating it.  And if passed in with a value other than LLDB_INVALID_ADDRESS,
    ///     this should point to an already allocated structure with the values already written.
    ///
    /// @param[in] errors
    ///     Errors will be written here if there are any.
    ///
    /// @param[in] stop_others
    ///     If \b true, run only this thread, if \b false let all threads run.
    ///
    /// @param[in] single_thread_timeout_usec
    ///     If \b true, run only this thread, if \b false let all threads run.
    ///
    /// @param[in] try_all_threads
    ///     If \b true, run only this thread, if \b false let all threads run.
    ///
    /// @param[out] results
    ///     The result value will be put here after running the function.
    ///
    /// @return
    ///     Returns one of the ExecutionResults enum indicating function call status.
    //------------------------------------------------------------------
    ExecutionResults ExecuteFunction(ExecutionContext &context, lldb::addr_t *args_addr_ptr, Stream &errors, bool stop_others, uint32_t single_thread_timeout_usec, bool try_all_threads, Value &results);
    ExecutionResults ExecuteFunctionWithABI(ExecutionContext &context, Stream &errors, Value &results);

    ThreadPlan *GetThreadPlanToCallFunction (ExecutionContext &exc_context, lldb::addr_t &args_addr_ref, Stream &errors, bool stop_others, bool discard_on_error = true);
    bool FetchFunctionResults (ExecutionContext &exc_context, lldb::addr_t args_addr, Value &ret_value);
    void DeallocateFunctionResults (ExecutionContext &exc_context, lldb::addr_t args_addr);
        
protected:
    //------------------------------------------------------------------
    // Classes that inherit from ClangFunction can see and modify these
    //------------------------------------------------------------------

private:
	//------------------------------------------------------------------
	// For ClangFunction only
	//------------------------------------------------------------------
    
   Function *m_function_ptr; // The function we're going to call.  May be NULL if we don't have debug info for the function.
   Address    m_function_addr; // If we don't have the FunctionSP, we at least need the address & return type.
   void *m_function_return_qual_type;  // The opaque clang qual type for the function return type.
   ClangASTContext *m_clang_ast_context;  // This is the clang_ast_context that we're getting types from the and value, and the function return the function pointer is NULL.

   std::string m_wrapper_function_name;
   std::string m_wrapper_struct_name;
   const clang::ASTRecordLayout *m_struct_layout;
   ValueList m_arg_values;
   lldb::addr_t m_wrapper_fun_addr;
   std::list<lldb::addr_t> m_wrapper_args_addrs;
   
   size_t m_value_struct_size;
   size_t m_return_offset;
   uint64_t m_return_size;  // Not strictly necessary, could get it from the Function...
   bool m_compiled;
   bool m_JITted;
};

} // Namespace lldb_private
#endif  // lldb_ClangFunction_h_
