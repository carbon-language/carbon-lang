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
        
        if (AppleObjCRuntime::GetObjCVersion (process, objc_module_sp) == lldb::eAppleObjC_V2)
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

ClangUtilityFunction *
AppleObjCRuntimeV2::CreateObjectChecker(const char *name)
{
    char check_function_code[1024];
    
    int len = 0;
    if (m_has_object_getClass)
    {
        len = ::snprintf (check_function_code, 
                          sizeof(check_function_code),
                          "extern \"C\" void *gdb_object_getClass(void *);    \n"
                          "extern \"C\" void                                  \n"
                          "%s(void *$__lldb_arg_obj)                          \n"
                          "{                                                  \n"
                          "   if ($__lldb_arg_obj == (void *)0)               \n"
                          "       return; // nil is ok                        \n" 
                          "   if (!gdb_object_getClass($__lldb_arg_obj))      \n"
                          "       *((volatile int *)0) = 'ocgc';              \n"
                          "}                                                  \n",
                          name);
    }
    else
    {
        len = ::snprintf (check_function_code, 
                          sizeof(check_function_code), 
                          "extern \"C\" void *gdb_class_getClass(void *);       \n"
                          "extern \"C\" void                                    \n"
                          "%s(void *$__lldb_arg_obj)                            \n"
                          "{                                                    \n"
                          "   if ($__lldb_arg_obj == (void *)0)                 \n"
                          "       return; // nil is ok                          \n" 
                          "    void **$isa_ptr = (void **)$__lldb_arg_obj;      \n"
                          "    if (*$isa_ptr == (void *)0 || !gdb_class_getClass(*$isa_ptr)) \n"
                          "       *((volatile int *)0) = 'ocgc';                \n"
                          "}                                                    \n", 
                          name);
    }
    
    assert (len < sizeof(check_function_code));

    return new ClangUtilityFunction(check_function_code, name);
}
