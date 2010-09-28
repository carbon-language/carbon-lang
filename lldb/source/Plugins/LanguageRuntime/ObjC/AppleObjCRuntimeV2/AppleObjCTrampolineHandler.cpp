//===-- AppleObjCTrampolineHandler.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleObjCTrampolineHandler.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "AppleThreadPlanStepThroughObjCTrampoline.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"

using namespace lldb;
using namespace lldb_private;

const AppleObjCTrampolineHandler::DispatchFunction
AppleObjCTrampolineHandler::g_dispatch_functions[] =
{
    // NAME                              STRET  SUPER  FIXUP TYPE
    {"objc_msgSend",                     false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSend_fixup",               false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_fixedup",             false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {"objc_msgSend_stret",               true,  false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSend_stret_fixup",         true,  false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_stret_fixedup",       true,  false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {"objc_msgSend_fpret",               false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSend_fpret_fixup",         false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_fpret_fixedup",       false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {"objc_msgSend_fp2ret",              false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSend_fp2ret_fixup",        false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_fp2ret_fixedup",      false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {"objc_msgSendSuper",                false, true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper_stret",          true,  true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper2",               false, true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper2_fixup",         false, true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSendSuper2_fixedup",       false, true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {"objc_msgSendSuper2_stret",         true,  true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper2_stret_fixup",   true,  true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSendSuper2_stret_fixedup", true,  true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {NULL}
};

bool
AppleObjCTrampolineHandler::ModuleIsObjCLibrary (const ModuleSP &module_sp)
{
    const FileSpec &module_file_spec = module_sp->GetFileSpec();
    static ConstString ObjCName ("libobjc.A.dylib");
    
    if (module_file_spec)
    {
        if (module_file_spec.GetFilename() == ObjCName)
            return true;
    }
    
    return false;
}

AppleObjCTrampolineHandler::AppleObjCTrampolineHandler (ProcessSP process_sp, ModuleSP objc_module) :
    m_process_sp (process_sp),
    m_objc_module_sp (objc_module),
    m_impl_fn_addr (LLDB_INVALID_ADDRESS),
    m_impl_stret_fn_addr (LLDB_INVALID_ADDRESS)
{
    // Look up the known resolution functions:
    
    ConstString get_impl_name("class_getMethodImplementation");
    ConstString get_impl_stret_name("class_getMethodImplementation_stret");
    
    Target *target = m_process_sp ? &m_process_sp->GetTarget() : NULL;
    const Symbol *class_getMethodImplementation = m_objc_module_sp->FindFirstSymbolWithNameAndType (get_impl_name, eSymbolTypeCode);
    const Symbol *class_getMethodImplementation_stret = m_objc_module_sp->FindFirstSymbolWithNameAndType (get_impl_stret_name, eSymbolTypeCode);
    
    if (class_getMethodImplementation)
        m_impl_fn_addr = class_getMethodImplementation->GetValue().GetLoadAddress(target);
    if  (class_getMethodImplementation_stret)
        m_impl_stret_fn_addr = class_getMethodImplementation_stret->GetValue().GetLoadAddress(target);
    
    // FIXME: Do some kind of logging here.
    if (m_impl_fn_addr == LLDB_INVALID_ADDRESS || m_impl_stret_fn_addr == LLDB_INVALID_ADDRESS)
        return;
        
    // Look up the addresses for the objc dispatch functions and cache them.  For now I'm inspecting the symbol
    // names dynamically to figure out how to dispatch to them.  If it becomes more complicated than this we can 
    // turn the g_dispatch_functions char * array into a template table, and populate the DispatchFunction map
    // from there.

    for (int i = 0; g_dispatch_functions[i].name != NULL; i++)
    {
        ConstString name_const_str(g_dispatch_functions[i].name);
        const Symbol *msgSend_symbol = m_objc_module_sp->FindFirstSymbolWithNameAndType (name_const_str, eSymbolTypeCode);
        if (msgSend_symbol)
        {
            // FixMe: Make g_dispatch_functions static table of DisptachFunctions, and have the map be address->index.
            // Problem is we also need to lookup the dispatch function.  For now we could have a side table of stret & non-stret
            // dispatch functions.  If that's as complex as it gets, we're fine.
            
            lldb::addr_t sym_addr = msgSend_symbol->GetValue().GetLoadAddress(target);
            
            m_msgSend_map.insert(std::pair<lldb::addr_t, int>(sym_addr, i));
        }
    }
}

ThreadPlanSP
AppleObjCTrampolineHandler::GetStepThroughDispatchPlan (Thread &thread, bool stop_others)
{
    ThreadPlanSP ret_plan_sp;
    lldb::addr_t curr_pc = thread.GetRegisterContext()->GetPC();
    
    MsgsendMap::iterator pos;
    pos = m_msgSend_map.find (curr_pc);
    if (pos != m_msgSend_map.end())
    {
        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);

        const DispatchFunction *this_dispatch = &g_dispatch_functions[(*pos).second];
        
        lldb::StackFrameSP thread_cur_frame = thread.GetStackFrameAtIndex(0);
        
        Process *process = thread.CalculateProcess();
        const ABI *abi = process->GetABI();
        if (abi == NULL)
            return ret_plan_sp;
            
        Target *target = thread.CalculateTarget();
        
        // FIXME: Since neither the value nor the Clang QualType know their ASTContext, 
        // we have to make sure the type we put in our value list comes from the same ASTContext
        // the ABI will use to get the argument values.  THis is the bottom-most frame's module.

        ClangASTContext *clang_ast_context = target->GetScratchClangASTContext();
        ValueList argument_values;
        Value input_value;
        void *clang_void_ptr_type = clang_ast_context->GetVoidPtrType(false);
        input_value.SetValueType (Value::eValueTypeScalar);
        input_value.SetContext (Value::eContextTypeOpaqueClangQualType, clang_void_ptr_type);
        
        int obj_index;
        int sel_index;
        
        // If this is a struct return dispatch, then the first argument is the
        // return struct pointer, and the object is the second, and the selector is the third.
        // Otherwise the object is the first and the selector the second.
        if (this_dispatch->stret_return)
        {
            obj_index = 1;
            sel_index = 2;
            argument_values.PushValue(input_value);
            argument_values.PushValue(input_value);
            argument_values.PushValue(input_value);
        }
        else
        {
            obj_index = 0;
            sel_index = 1;
            argument_values.PushValue(input_value);
            argument_values.PushValue(input_value);
        }

        
        bool success = abi->GetArgumentValues (thread, argument_values);
        if (!success)
            return ret_plan_sp;
        
        // Okay, the first value here is the object, we actually want the class of that object.
        // For now we're just going with the ISA.  
        // FIXME: This should really be the return value of [object class] to properly handle KVO interposition.
        
        Value isa_value(*(argument_values.GetValueAtIndex(obj_index)));
        
        // This is a little cheesy, but since object->isa is the first field, 
        // making the object value a load address value and resolving it will get
        // the pointer sized data pointed to by that value...
        ExecutionContext exec_ctx;
        thread.Calculate (exec_ctx);

        isa_value.SetValueType(Value::eValueTypeLoadAddress);
        isa_value.ResolveValue(&exec_ctx, clang_ast_context->getASTContext());
        
        if (this_dispatch->fixedup == DispatchFunction::eFixUpFixed)
        {
            // For the FixedUp method the Selector is actually a pointer to a 
            // structure, the second field of which is the selector number.
            Value *sel_value = argument_values.GetValueAtIndex(sel_index);
            sel_value->GetScalar() += process->GetAddressByteSize();
            sel_value->SetValueType(Value::eValueTypeLoadAddress);
            sel_value->ResolveValue(&exec_ctx, clang_ast_context->getASTContext());            
        }
        else if (this_dispatch->fixedup == DispatchFunction::eFixUpToFix)
        {   
            // FIXME: If the method dispatch is not "fixed up" then the selector is actually a
            // pointer to the string name of the selector.  We need to look that up...
            // For now I'm going to punt on that and just return no plan.
            if (log)
                log->Printf ("Punting on stepping into un-fixed-up method dispatch.");
            return ret_plan_sp;
        }
        
        // FIXME: If this is a dispatch to the super-class, we need to get the super-class from
        // the class, and disaptch to that instead.
        // But for now I just punt and return no plan.
        if (this_dispatch->is_super)
        {   
            if (log)
                log->Printf ("Punting on stepping into super method dispatch.");
            return ret_plan_sp;
        }
        
        ValueList dispatch_values;
        dispatch_values.PushValue (isa_value);
        dispatch_values.PushValue(*(argument_values.GetValueAtIndex(sel_index)));
        
        if (log)
        {
            log->Printf("Resolving method call for class - 0x%llx and selector - 0x%llx",
                        dispatch_values.GetValueAtIndex(0)->GetScalar().ULongLong(),
                        dispatch_values.GetValueAtIndex(1)->GetScalar().ULongLong());
        }
        ObjCLanguageRuntime *objc_runtime = m_process_sp->GetObjCLanguageRuntime ();
        assert(objc_runtime != NULL);
        lldb::addr_t impl_addr = objc_runtime->LookupInMethodCache (dispatch_values.GetValueAtIndex(0)->GetScalar().ULongLong(),
                                                dispatch_values.GetValueAtIndex(1)->GetScalar().ULongLong());
                                                
        if (impl_addr == LLDB_INVALID_ADDRESS)
        {

            Address resolve_address(NULL, this_dispatch->stret_return ? m_impl_stret_fn_addr : m_impl_fn_addr);
            
            StreamString errors;
            { 
                // Scope for mutex locker:
                Mutex::Locker locker(m_impl_function_mutex);
                if (!m_impl_function.get())
                {
                     m_impl_function.reset(new ClangFunction(process->GetTargetTriple().GetCString(), 
                                                             clang_ast_context, 
                                                             clang_void_ptr_type, 
                                                             resolve_address, 
                                                             dispatch_values));
                            
                    unsigned num_errors = m_impl_function->CompileFunction(errors);
                    if (num_errors)
                    {
                        if (log)
                            log->Printf ("Error compiling function: \"%s\".", errors.GetData());
                        return ret_plan_sp;
                    }
                    
                    errors.Clear();
                    if (!m_impl_function->WriteFunctionWrapper(exec_ctx, errors))
                    {
                        if (log)
                            log->Printf ("Error Inserting function: \"%s\".", errors.GetData());
                        return ret_plan_sp;
                    }
                }
                
            }
            
            errors.Clear();
            
            // Now write down the argument values for this call.
            lldb::addr_t args_addr = LLDB_INVALID_ADDRESS;
            if (!m_impl_function->WriteFunctionArguments (exec_ctx, args_addr, resolve_address, dispatch_values, errors))
                return ret_plan_sp;
        
            ret_plan_sp.reset (new AppleThreadPlanStepThroughObjCTrampoline (thread, this, args_addr, 
                                                                        argument_values.GetValueAtIndex(0)->GetScalar().ULongLong(),
                                                                        dispatch_values.GetValueAtIndex(0)->GetScalar().ULongLong(),
                                                                        dispatch_values.GetValueAtIndex(1)->GetScalar().ULongLong(),
                                                                        stop_others));
        }
        else
        {
            if (log)
                log->Printf ("Found implementation address in cache: 0x%llx", impl_addr);
                 
            ret_plan_sp.reset (new ThreadPlanRunToAddress (thread, impl_addr, stop_others));
        }
    }
    
    return ret_plan_sp;
}

ClangFunction *
AppleObjCTrampolineHandler::GetLookupImplementationWrapperFunction ()
{
    return m_impl_function.get();
}
