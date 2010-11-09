//===-- AppleObjCRuntimeV2.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleObjCRuntimeV2.h"
#include "AppleObjCTrampolineHandler.h"

#include "llvm/Support/MachO.h"
#include "clang/AST/Type.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ClangUtilityFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

static const char *pluginName = "AppleObjCRuntimeV2";
static const char *pluginDesc = "Apple Objective C Language Runtime - Version 2";
static const char *pluginShort = "language.apple.objc.v2";

AppleObjCRuntimeV2::AppleObjCRuntimeV2 (Process *process, ModuleSP &objc_module_sp) : 
    lldb_private::AppleObjCRuntime (process)
{
    m_has_object_getClass = (objc_module_sp->FindFirstSymbolWithNameAndType(ConstString("gdb_object_getClass")) != NULL);
}

bool
AppleObjCRuntimeV2::GetObjectDescription (Stream &str, ValueObject &object, ExecutionContextScope *exe_scope)
{

    // ObjC objects can only be pointers:
    if (!object.IsPointerType())
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
    return GetObjectDescription(str, val, exe_scope);
                   
}
bool
AppleObjCRuntimeV2::GetObjectDescription (Stream &str, Value &value, ExecutionContextScope *exe_scope)
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
            str.Printf ("Value doesn't point to an ObjC object.\n");
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
        value.SetContext(Value::eContextTypeOpaqueClangQualType, opaque_type_ptr);    
    }

    ValueList arg_value_list;
    arg_value_list.PushValue(value);
    
    // This is the return value:
    const char *target_triple = exe_ctx.process->GetTargetTriple().GetCString();
    ClangASTContext *ast_context = exe_ctx.target->GetScratchClangASTContext();
    
    void *return_qualtype = ast_context->GetCStringType(true);
    Value ret;
    ret.SetContext(Value::eContextTypeOpaqueClangQualType, return_qualtype);
    
    // Now we're ready to call the function:
    ClangFunction func(target_triple, ast_context, return_qualtype, *function_address, arg_value_list);
    StreamString error_stream;
    
    lldb::addr_t wrapper_struct_addr = LLDB_INVALID_ADDRESS;
    func.InsertFunction(exe_ctx, wrapper_struct_addr, error_stream);

    const bool stop_others = true;
    const bool try_all_threads = true;
    const bool discard_on_error = true;
    
    ClangFunction::ExecutionResults results 
        = func.ExecuteFunction(exe_ctx, &wrapper_struct_addr, error_stream, stop_others, 1000, 
                               try_all_threads, discard_on_error, ret);
    if (results != ClangFunction::eExecutionCompleted)
    {
        str.Printf("Error evaluating Print Object function: %d.\n", results);
        return false;
    }
       
    addr_t result_ptr = ret.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
    
    // FIXME: poor man's strcpy - we should have a "read memory as string interface...
    
    Error error;
    std::vector<char> desc;
    while (1)
    {
        char byte = '\0';
        if (exe_ctx.process->ReadMemory(result_ptr + desc.size(), &byte, 1, error) != 1)
            break;
        
        desc.push_back(byte);

        if (byte == '\0')
            break;
    }
    
    if (!desc.empty())
    {
        str.PutCString(&desc.front());
        return true;
    }
    return false;

}

lldb::ValueObjectSP
AppleObjCRuntimeV2::GetDynamicValue (lldb::ValueObjectSP in_value, ExecutionContextScope *exe_scope)
{
    lldb::ValueObjectSP ret_sp;
    return ret_sp;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
lldb_private::LanguageRuntime *
AppleObjCRuntimeV2::CreateInstance (Process *process, lldb::LanguageType language)
{
    // FIXME: This should be a MacOS or iOS process, and we need to look for the OBJC section to make
    // sure we aren't using the V1 runtime.
    if (language == eLanguageTypeObjC)
    {
        ModuleSP objc_module_sp;
        
        if (AppleObjCRuntime::GetObjCVersion (process, objc_module_sp) == AppleObjCRuntime::eObjC_V2)
            return new AppleObjCRuntimeV2 (process, objc_module_sp);
        else
            return NULL;
    }
    else
        return NULL;
}

void
AppleObjCRuntimeV2::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);    
}

void
AppleObjCRuntimeV2::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
AppleObjCRuntimeV2::GetPluginName()
{
    return pluginName;
}

const char *
AppleObjCRuntimeV2::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
AppleObjCRuntimeV2::GetPluginVersion()
{
    return 1;
}

void
AppleObjCRuntimeV2::SetExceptionBreakpoints ()
{
    if (!m_process)
        return;
        
    if (!m_objc_exception_bp_sp)
    {
        m_objc_exception_bp_sp = m_process->GetTarget().CreateBreakpoint (NULL,
                                                                          "__cxa_throw",
                                                                          eFunctionNameTypeBase, 
                                                                          true);
    }
}

struct BufStruct {
    char contents[1024];
};

ClangUtilityFunction *
AppleObjCRuntimeV2::CreateObjectChecker(const char *name)
{
    std::auto_ptr<BufStruct> buf(new BufStruct);
    
    if (m_has_object_getClass)
    {
        assert(snprintf(&buf->contents[0], sizeof(buf->contents),
                        "extern \"C\" int gdb_object_getClass(void *);      \n"
                        "extern \"C\" void                                  \n"
                        "%s(void *$__lldb_arg_obj)                          \n"
                        "{                                                  \n"
                        "   if (!gdb_object_getClass($__lldb_arg_obj))      \n"
                        "       abort();                                    \n"
                        "}                                                  \n",
                        name) < sizeof(buf->contents));
    }
    else
    {
        assert(snprintf(&buf->contents[0], sizeof(buf->contents), 
                        "extern \"C\" int gdb_class_getClass(void *);         \n"
                        "extern \"C\" void                                    \n"
                        "%s(void *$__lldb_arg_obj)                            \n"
                        "{                                                    \n"
                        "    void **$isa_ptr = (void **)$__lldb_arg_obj;      \n"
                        "    if (!$isa_ptr || !gdb_class_getClass(*$isa_ptr)) \n"
                        "        abort();                                     \n"
                        "}                                                    \n", 
                        name) < sizeof(buf->contents));
    }
    
    return new ClangUtilityFunction(buf->contents, name);
}
