//===-- AppleGetPendingItemsHandler.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleGetPendingItemsHandler.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ClangUtilityFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

const char *AppleGetPendingItemsHandler::g_get_pending_items_function_name = "__lldb_backtrace_recording_get_pending_items";
const char *AppleGetPendingItemsHandler::g_get_pending_items_function_code = "                                  \n\
extern \"C\"                                                                                                    \n\
{                                                                                                               \n\
    /*                                                                                                          \n\
     * mach defines                                                                                             \n\
     */                                                                                                         \n\
                                                                                                                \n\
    typedef unsigned int uint32_t;                                                                              \n\
    typedef unsigned long long uint64_t;                                                                        \n\
    typedef uint32_t mach_port_t;                                                                               \n\
    typedef mach_port_t vm_map_t;                                                                               \n\
    typedef int kern_return_t;                                                                                  \n\
    typedef uint64_t mach_vm_address_t;                                                                         \n\
    typedef uint64_t mach_vm_size_t;                                                                            \n\
                                                                                                                \n\
    mach_port_t mach_task_self ();                                                                              \n\
    kern_return_t mach_vm_deallocate (vm_map_t target, mach_vm_address_t address, mach_vm_size_t size);         \n\
                                                                                                                \n\
    /*                                                                                                          \n\
     * libBacktraceRecording defines                                                                            \n\
     */                                                                                                         \n\
                                                                                                                \n\
    typedef uint32_t queue_list_scope_t;                                                                        \n\
    typedef void *dispatch_queue_t;                                                                             \n\
    typedef void *introspection_dispatch_queue_info_t;                                                          \n\
    typedef void *introspection_dispatch_item_info_ref;                                                         \n\
                                                                                                                \n\
    extern uint64_t __introspection_dispatch_queue_get_pending_items (dispatch_queue_t queue,                   \n\
                                                 introspection_dispatch_item_info_ref *returned_queues_buffer,  \n\
                                                 uint64_t *returned_queues_buffer_size);                        \n\
    extern int printf(const char *format, ...);                                                                 \n\
                                                                                                                \n\
    /*                                                                                                          \n\
     * return type define                                                                                       \n\
     */                                                                                                         \n\
                                                                                                                \n\
    struct get_pending_items_return_values                                                                      \n\
    {                                                                                                           \n\
        uint64_t pending_items_buffer_ptr;    /* the address of the items buffer from libBacktraceRecording */  \n\
        uint64_t pending_items_buffer_size;   /* the size of the items buffer from libBacktraceRecording */     \n\
        uint64_t count;                /* the number of items included in the queues buffer */                  \n\
    };                                                                                                          \n\
                                                                                                                \n\
    void  __lldb_backtrace_recording_get_pending_items                                                          \n\
                                               (struct get_pending_items_return_values *return_buffer,          \n\
                                                int debug,                                                      \n\
                                                uint64_t /* dispatch_queue_t */ queue,                          \n\
                                                void *page_to_free,                                             \n\
                                                uint64_t page_to_free_size)                                     \n\
{                                                                                                               \n\
    if (debug)                                                                                                  \n\
      printf (\"entering get_pending_items with args return_buffer == %p, debug == %d, queue == 0x%llx, page_to_free == %p, page_to_free_size == 0x%llx\\n\", return_buffer, debug, queue, page_to_free, page_to_free_size); \n\
    if (page_to_free != 0)                                                                                      \n\
    {                                                                                                           \n\
        mach_vm_deallocate (mach_task_self(), (mach_vm_address_t) page_to_free, (mach_vm_size_t) page_to_free_size); \n\
    }                                                                                                           \n\
                                                                                                                \n\
    return_buffer->count = __introspection_dispatch_queue_get_pending_items (                                   \n\
                                                      (void*) queue,                                            \n\
                                                      (void**)&return_buffer->pending_items_buffer_ptr,         \n\
                                                      &return_buffer->pending_items_buffer_size);               \n\
    if (debug)                                                                                                  \n\
        printf(\"result was count %lld\\n\", return_buffer->count);                                             \n\
}                                                                                                               \n\
}                                                                                                               \n\
";

AppleGetPendingItemsHandler::AppleGetPendingItemsHandler (Process *process) :
    m_process (process),
    m_get_pending_items_function (),
    m_get_pending_items_impl_code (),
    m_get_pending_items_function_mutex(),
    m_get_pending_items_return_buffer_addr (LLDB_INVALID_ADDRESS),
    m_get_pending_items_retbuffer_mutex()
{
}

AppleGetPendingItemsHandler::~AppleGetPendingItemsHandler ()
{
}

void
AppleGetPendingItemsHandler::Detach ()
{

    if (m_process && m_process->IsAlive() && m_get_pending_items_return_buffer_addr != LLDB_INVALID_ADDRESS)
    {
        Mutex::Locker locker;
        locker.TryLock (m_get_pending_items_retbuffer_mutex);  // Even if we don't get the lock, deallocate the buffer
        m_process->DeallocateMemory (m_get_pending_items_return_buffer_addr);
    }
}

// Compile our __lldb_backtrace_recording_get_pending_items() function (from the
// source above in g_get_pending_items_function_code) if we don't find that function in the inferior
// already with USE_BUILTIN_FUNCTION defined.  (e.g. this would be the case for testing.)
//
// Insert the __lldb_backtrace_recording_get_pending_items into the inferior process if needed.
//
// Write the get_pending_items_arglist into the inferior's memory space to prepare for the call.
// 
// Returns the address of the arguments written down in the inferior process, which can be used to
// make the function call.

lldb::addr_t
AppleGetPendingItemsHandler::SetupGetPendingItemsFunction (Thread &thread, ValueList &get_pending_items_arglist)
{
    ExecutionContext exe_ctx (thread.shared_from_this());
    Address impl_code_address;
    StreamString errors;
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SYSTEM_RUNTIME));
    lldb::addr_t args_addr = LLDB_INVALID_ADDRESS;

    // Scope for mutex locker:
    {
        Mutex::Locker locker(m_get_pending_items_function_mutex);
        
        // First stage is to make the ClangUtility to hold our injected function:

#define USE_BUILTIN_FUNCTION 0  // Define this to 1 and we will use the get_implementation function found in the target.
                                // This is useful for debugging additions to the get_impl function 'cause you don't have
                                // to bother with string-ifying the code into g_get_pending_items_function_code.
        
        if (USE_BUILTIN_FUNCTION)
        {
            ConstString our_utility_function_name("__lldb_backtrace_recording_get_pending_items");
            SymbolContextList sc_list;
            
            exe_ctx.GetTargetRef().GetImages().FindSymbolsWithNameAndType (our_utility_function_name, eSymbolTypeCode, sc_list);
            if (sc_list.GetSize() == 1)
            {
                SymbolContext sc;
                sc_list.GetContextAtIndex(0, sc);
                if (sc.symbol != NULL)
                    impl_code_address = sc.symbol->GetAddress();
                    
                //lldb::addr_t addr = impl_code_address.GetOpcodeLoadAddress (exe_ctx.GetTargetPtr());
                //printf ("Getting address for our_utility_function: 0x%" PRIx64 ".\n", addr);
            }
            else
            {
                //printf ("Could not find queues introspection function address.\n");
                return args_addr;
            }
        }
        else if (!m_get_pending_items_impl_code.get())
        {
            if (g_get_pending_items_function_code != NULL)
            {
                m_get_pending_items_impl_code.reset (new ClangUtilityFunction (g_get_pending_items_function_code,
                                                             g_get_pending_items_function_name));
                if (!m_get_pending_items_impl_code->Install(errors, exe_ctx))
                {
                    if (log)
                        log->Printf ("Failed to install pending-items introspection: %s.", errors.GetData());
                    m_get_pending_items_impl_code.reset();
                    return args_addr;
                }
            }
            else
            {
                if (log)
                    log->Printf("No pending-items introspection code found.");
                errors.Printf ("No pending-items introspection code found.");
                return LLDB_INVALID_ADDRESS;
            }
            
            impl_code_address.Clear();
            impl_code_address.SetOffset(m_get_pending_items_impl_code->StartAddress());
        }
        else
        {
            impl_code_address.Clear();
            impl_code_address.SetOffset(m_get_pending_items_impl_code->StartAddress());
        }

        // Next make the runner function for our implementation utility function.
        if (!m_get_pending_items_function.get())
        {
            ClangASTContext *clang_ast_context = thread.GetProcess()->GetTarget().GetScratchClangASTContext();
            ClangASTType get_pending_items_return_type = clang_ast_context->GetBasicType(eBasicTypeVoid).GetPointerType();
            m_get_pending_items_function.reset(new ClangFunction (thread,
                                                     get_pending_items_return_type,
                                                     impl_code_address,
                                                     get_pending_items_arglist));
            
            errors.Clear();        
            unsigned num_errors = m_get_pending_items_function->CompileFunction(errors);
            if (num_errors)
            {
                if (log)
                    log->Printf ("Error compiling pending-items function: \"%s\".", errors.GetData());
                return args_addr;
            }
            
            errors.Clear();
            if (!m_get_pending_items_function->WriteFunctionWrapper(exe_ctx, errors))
            {
                if (log)
                    log->Printf ("Error Inserting pending-items function: \"%s\".", errors.GetData());
                return args_addr;
            }
        }
    }
    
    errors.Clear();
    
    // Now write down the argument values for this particular call.  This looks like it might be a race condition
    // if other threads were calling into here, but actually it isn't because we allocate a new args structure for
    // this call by passing args_addr = LLDB_INVALID_ADDRESS...

    if (!m_get_pending_items_function->WriteFunctionArguments (exe_ctx, args_addr, impl_code_address, get_pending_items_arglist, errors))
    {
        if (log)
            log->Printf ("Error writing pending-items function arguments: \"%s\".", errors.GetData());
        return args_addr;
    }
        
    return args_addr;
}

AppleGetPendingItemsHandler::GetPendingItemsReturnInfo
AppleGetPendingItemsHandler::GetPendingItems (Thread &thread, addr_t queue, addr_t page_to_free, uint64_t page_to_free_size, Error &error)
{
    lldb::StackFrameSP thread_cur_frame = thread.GetStackFrameAtIndex(0);
    ProcessSP process_sp (thread.CalculateProcess());
    TargetSP target_sp (thread.CalculateTarget());
    ClangASTContext *clang_ast_context = target_sp->GetScratchClangASTContext();
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SYSTEM_RUNTIME));

    GetPendingItemsReturnInfo return_value;
    return_value.items_buffer_ptr = LLDB_INVALID_ADDRESS;
    return_value.items_buffer_size = 0;
    return_value.count = 0;

    error.Clear();

    // Set up the arguments for a call to

    // struct get_pending_items_return_values
    // {
    //     uint64_t pending_items_buffer_ptr;    /* the address of the items buffer from libBacktraceRecording */
    //     uint64_t pending_items_buffer_size;   /* the size of the items buffer from libBacktraceRecording */
    //     uint64_t count;                /* the number of items included in the queues buffer */
    // };
    //
    // void  __lldb_backtrace_recording_get_pending_items
    //                                            (struct get_pending_items_return_values *return_buffer,
    //                                             int debug,
    //                                             uint64_t /* dispatch_queue_t */ queue
    //                                             void *page_to_free,
    //                                             uint64_t page_to_free_size)

    // Where the return_buffer argument points to a 24 byte region of memory already allocated by lldb in
    // the inferior process.

    ClangASTType clang_void_ptr_type = clang_ast_context->GetBasicType(eBasicTypeVoid).GetPointerType();
    Value return_buffer_ptr_value;
    return_buffer_ptr_value.SetValueType (Value::eValueTypeScalar);
    return_buffer_ptr_value.SetClangType (clang_void_ptr_type);

    ClangASTType clang_int_type = clang_ast_context->GetBasicType(eBasicTypeInt);
    Value debug_value;
    debug_value.SetValueType (Value::eValueTypeScalar);
    debug_value.SetClangType (clang_int_type);

    ClangASTType clang_uint64_type = clang_ast_context->GetBasicType(eBasicTypeUnsignedLongLong);
    Value queue_value;
    queue_value.SetValueType (Value::eValueTypeScalar);
    queue_value.SetClangType (clang_uint64_type);

    Value page_to_free_value;
    page_to_free_value.SetValueType (Value::eValueTypeScalar);
    page_to_free_value.SetClangType (clang_void_ptr_type);

    Value page_to_free_size_value;
    page_to_free_size_value.SetValueType (Value::eValueTypeScalar);
    page_to_free_size_value.SetClangType (clang_uint64_type);


    Mutex::Locker locker(m_get_pending_items_retbuffer_mutex);
    if (m_get_pending_items_return_buffer_addr == LLDB_INVALID_ADDRESS)
    {
        addr_t bufaddr = process_sp->AllocateMemory (32, ePermissionsReadable | ePermissionsWritable, error);
        if (!error.Success() || bufaddr == LLDB_INVALID_ADDRESS)
        {
            if (log)
                log->Printf ("Failed to allocate memory for return buffer for get current queues func call");
            return return_value;
        }
        m_get_pending_items_return_buffer_addr = bufaddr;
    }

    ValueList argument_values;

    return_buffer_ptr_value.GetScalar() = m_get_pending_items_return_buffer_addr;
    argument_values.PushValue (return_buffer_ptr_value);

    debug_value.GetScalar() = 0;
    argument_values.PushValue (debug_value);

    queue_value.GetScalar() = queue;
    argument_values.PushValue (queue_value);

    if (page_to_free != LLDB_INVALID_ADDRESS)
        page_to_free_value.GetScalar() = page_to_free;
    else
        page_to_free_value.GetScalar() = 0;
    argument_values.PushValue (page_to_free_value);

    page_to_free_size_value.GetScalar() = page_to_free_size;
    argument_values.PushValue (page_to_free_size_value);

    addr_t args_addr = SetupGetPendingItemsFunction (thread, argument_values);

    StreamString errors;
    ExecutionContext exe_ctx;
    EvaluateExpressionOptions options;
    options.SetUnwindOnError (true);
    options.SetIgnoreBreakpoints (true);
    options.SetStopOthers (true);
    thread.CalculateExecutionContext (exe_ctx);

    if (m_get_pending_items_function == NULL)
    {
        error.SetErrorString ("Unable to compile function to call __introspection_dispatch_queue_get_pending_items");
    }


    ExecutionResults func_call_ret;
    Value results;
    func_call_ret =  m_get_pending_items_function->ExecuteFunction (exe_ctx, &args_addr, options, errors, results);
    if (func_call_ret != eExecutionCompleted || !error.Success())
    {
        if (log)
            log->Printf ("Unable to call __introspection_dispatch_queue_get_pending_items(), got ExecutionResults %d, error contains %s", func_call_ret, error.AsCString(""));
        error.SetErrorString ("Unable to call __introspection_dispatch_queue_get_pending_items() for list of queues");
        return return_value;
    }

    return_value.items_buffer_ptr = m_process->ReadUnsignedIntegerFromMemory (m_get_pending_items_return_buffer_addr, 8, LLDB_INVALID_ADDRESS, error);
    if (!error.Success() || return_value.items_buffer_ptr == LLDB_INVALID_ADDRESS)
    {
        return_value.items_buffer_ptr = LLDB_INVALID_ADDRESS;
        return return_value;
    }

    return_value.items_buffer_size = m_process->ReadUnsignedIntegerFromMemory (m_get_pending_items_return_buffer_addr + 8, 8, 0, error);

    if (!error.Success())
    {
        return_value.items_buffer_ptr = LLDB_INVALID_ADDRESS;
        return return_value;
    }

    return_value.count = m_process->ReadUnsignedIntegerFromMemory (m_get_pending_items_return_buffer_addr + 16, 8, 0, error);
    if (!error.Success())
    {
        return_value.items_buffer_ptr = LLDB_INVALID_ADDRESS;
        return return_value;
    }

    if (log)
        log->Printf ("AppleGetPendingItemsHandler called __introspection_dispatch_queue_get_pending_items (page_to_free == 0x%" PRIx64 ", size = %" PRId64 "), returned page is at 0x%" PRIx64 ", size %" PRId64 ", count = %" PRId64, page_to_free, page_to_free_size, return_value.items_buffer_ptr, return_value.items_buffer_ptr, return_value.count);

    return return_value;
}
