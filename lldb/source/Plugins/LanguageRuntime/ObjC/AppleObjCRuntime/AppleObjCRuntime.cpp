//===-- AppleObjCRuntime.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleObjCRuntime.h"
#include "AppleObjCTrampolineHandler.h"

#include "llvm/Support/MachO.h"
#include "clang/AST/Type.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

bool
AppleObjCRuntime::GetObjectDescription (Stream &str, ValueObject &object)
{
    bool is_signed;
    // ObjC objects can only be pointers, but we extend this to integer types because an expression might just
    // result in an address, and we should try that to see if the address is an ObjC object.
    
    if (!(object.IsPointerType() || object.IsIntegerType(is_signed)))
        return NULL;
    
    // Make the argument list: we pass one arg, the address of our pointer, to the print function.
    Scalar scalar;
    
    if (!ClangASTType::GetValueAsScalar (object.GetClangAST(),
                                        object.GetClangType(),
                                        object.GetDataExtractor(),
                                        0,
                                        object.GetByteSize(),
                                        scalar))
        return NULL;
                        
    Value val(scalar);                   
    return GetObjectDescription(str, val, object.GetExecutionContextScope());
                   
}
bool
AppleObjCRuntime::GetObjectDescription (Stream &strm, Value &value, ExecutionContextScope *exe_scope)
{
    if (!m_read_objc_library)
        return false;
        
    ExecutionContext exe_ctx;
    exe_scope->CalculateExecutionContext(exe_ctx);
    
    if (!exe_ctx.process)
        return false;
    
    // We need other parts of the exe_ctx, but the processes have to match.
    assert (m_process == exe_ctx.process);
    
    // Get the function address for the print function.
    const Address *function_address = GetPrintForDebuggerAddr();
    if (!function_address)
        return false;
    
    if (value.GetClangType())
    {
        clang::QualType value_type = clang::QualType::getFromOpaquePtr (value.GetClangType());
        if (!value_type->isObjCObjectPointerType())
        {
            strm.Printf ("Value doesn't point to an ObjC object.\n");
            return false;
        }
    }
    else 
    {
        // If it is not a pointer, see if we can make it into a pointer.
        ClangASTContext *ast_context = exe_ctx.target->GetScratchClangASTContext();
        void *opaque_type_ptr = ast_context->GetBuiltInType_objc_id();
        if (opaque_type_ptr == NULL)
            opaque_type_ptr = ast_context->GetVoidPtrType(false);
        value.SetContext(Value::eContextTypeClangType, opaque_type_ptr);    
    }

    ValueList arg_value_list;
    arg_value_list.PushValue(value);
    
    // This is the return value:
    ClangASTContext *ast_context = exe_ctx.target->GetScratchClangASTContext();
    
    void *return_qualtype = ast_context->GetCStringType(true);
    Value ret;
    ret.SetContext(Value::eContextTypeClangType, return_qualtype);
    
    // Now we're ready to call the function:
    ClangFunction func (*exe_ctx.GetBestExecutionContextScope(),
                        ast_context, 
                        return_qualtype, 
                        *function_address, 
                        arg_value_list);

    StreamString error_stream;
    
    lldb::addr_t wrapper_struct_addr = LLDB_INVALID_ADDRESS;
    func.InsertFunction(exe_ctx, wrapper_struct_addr, error_stream);

    bool unwind_on_error = true;
    bool try_all_threads = true;
    bool stop_others = true;
    
    ExecutionResults results = func.ExecuteFunction (exe_ctx, 
                                                     &wrapper_struct_addr, 
                                                     error_stream, 
                                                     stop_others, 
                                                     1000000, 
                                                     try_all_threads, 
                                                     unwind_on_error, 
                                                     ret);
    if (results != eExecutionCompleted)
    {
        strm.Printf("Error evaluating Print Object function: %d.\n", results);
        return false;
    }
       
    addr_t result_ptr = ret.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
    
    char buf[512];
    size_t cstr_len = 0;    
    size_t full_buffer_len = sizeof (buf) - 1;
    size_t curr_len = full_buffer_len;
    while (curr_len == full_buffer_len)
    {
        curr_len = exe_ctx.process->ReadCStringFromMemory(result_ptr + cstr_len, buf, sizeof(buf));
        strm.Write (buf, curr_len);
        cstr_len += curr_len;
    }
    return cstr_len > 0;
}

Address *
AppleObjCRuntime::GetPrintForDebuggerAddr()
{
    if (!m_PrintForDebugger_addr.get())
    {
        ModuleList &modules = m_process->GetTarget().GetImages();
        
        SymbolContextList contexts;
        SymbolContext context;
        
        if ((!modules.FindSymbolsWithNameAndType(ConstString ("_NSPrintForDebugger"), eSymbolTypeCode, contexts)) &&
           (!modules.FindSymbolsWithNameAndType(ConstString ("_CFPrintForDebugger"), eSymbolTypeCode, contexts)))
            return NULL;
        
        contexts.GetContextAtIndex(0, context);
        
        m_PrintForDebugger_addr.reset(new Address(context.symbol->GetValue()));
    }
    
    return m_PrintForDebugger_addr.get();
}

bool
AppleObjCRuntime::CouldHaveDynamicValue (ValueObject &in_value)
{
    lldb::LanguageType known_type = in_value.GetObjectRuntimeLanguage();
    if (known_type == lldb::eLanguageTypeObjC)
        return true;
    else
        return in_value.IsPointerType();
}

bool
AppleObjCRuntime::GetDynamicTypeAndAddress (ValueObject &in_value, 
                                            lldb::DynamicValueType use_dynamic, 
                                            TypeAndOrName &class_type_or_name, 
                                            Address &address)
{
    return false;
}

bool
AppleObjCRuntime::AppleIsModuleObjCLibrary (const ModuleSP &module_sp)
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

bool
AppleObjCRuntime::IsModuleObjCLibrary (const ModuleSP &module_sp)
{
    return AppleIsModuleObjCLibrary(module_sp);
}

bool
AppleObjCRuntime::ReadObjCLibrary (const ModuleSP &module_sp)
{
    // Maybe check here and if we have a handler already, and the UUID of this module is the same as the one in the
    // current module, then we don't have to reread it?
    m_objc_trampoline_handler_ap.reset(new AppleObjCTrampolineHandler (m_process->GetSP(), module_sp));
    if (m_objc_trampoline_handler_ap.get() != NULL)
    {
        m_read_objc_library = true;
        return true;
    }
    else
        return false;
}

ThreadPlanSP
AppleObjCRuntime::GetStepThroughTrampolinePlan (Thread &thread, bool stop_others)
{
    ThreadPlanSP thread_plan_sp;
    if (m_objc_trampoline_handler_ap.get())
        thread_plan_sp = m_objc_trampoline_handler_ap->GetStepThroughDispatchPlan (thread, stop_others);
    return thread_plan_sp;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
enum ObjCRuntimeVersions
AppleObjCRuntime::GetObjCVersion (Process *process, ModuleSP &objc_module_sp)
{
    ModuleList &images = process->GetTarget().GetImages();
    size_t num_images = images.GetSize();
    for (size_t i = 0; i < num_images; i++)
    {
        ModuleSP module_sp = images.GetModuleAtIndex(i);
        if (AppleIsModuleObjCLibrary (module_sp))
        {
            objc_module_sp = module_sp;
            ObjectFile *ofile = module_sp->GetObjectFile();
            if (!ofile)
                return eObjC_VersionUnknown;
            
            SectionList *sections = ofile->GetSectionList();
            if (!sections)
                return eObjC_VersionUnknown;    
            SectionSP v1_telltale_section_sp = sections->FindSectionByName(ConstString ("__OBJC"));
            if (v1_telltale_section_sp)
            {
                return eAppleObjC_V1;
            }
            return eAppleObjC_V2;
        }
    }
            
    return eObjC_VersionUnknown;
}

void
AppleObjCRuntime::ClearExceptionBreakpoints ()
{
    if (!m_process)
        return;
    
    if (m_objc_exception_bp_sp.get())
    {
        m_objc_exception_bp_sp->SetEnabled (false);
    }
}

bool
AppleObjCRuntime::ExceptionBreakpointsExplainStop (lldb::StopInfoSP stop_reason)
{
    if (!m_process)
        return false;
    
    if (!stop_reason || 
        stop_reason->GetStopReason() != eStopReasonBreakpoint)
        return false;
    
    uint64_t break_site_id = stop_reason->GetValue();
    lldb::BreakpointSiteSP bp_site_sp = m_process->GetBreakpointSiteList().FindByID(break_site_id);
    
    if (!bp_site_sp)
        return false;
    
    uint32_t num_owners = bp_site_sp->GetNumberOfOwners();
    
    bool        check_objc_exception = false;
    break_id_t  objc_exception_bid;
    
    if (m_objc_exception_bp_sp)
    {
        check_objc_exception = true;
        objc_exception_bid = m_objc_exception_bp_sp->GetID();
    }
    
    for (uint32_t i = 0; i < num_owners; i++)
    {
        break_id_t bid = bp_site_sp->GetOwnerAtIndex(i)->GetBreakpoint().GetID();
        
        if ((check_objc_exception && (bid == objc_exception_bid)))
            return true;
    }
    
    return false;
}
