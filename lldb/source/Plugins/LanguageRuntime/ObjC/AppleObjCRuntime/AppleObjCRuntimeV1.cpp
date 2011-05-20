//===-- AppleObjCRuntimeV1.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleObjCRuntimeV1.h"
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

static const char *pluginName = "AppleObjCRuntimeV1";
static const char *pluginDesc = "Apple Objective C Language Runtime - Version 1";
static const char *pluginShort = "language.apple.objc.v1";

bool
AppleObjCRuntimeV1::GetDynamicTypeAndAddress (ValueObject &in_value, 
                                             lldb::DynamicValueType use_dynamic, 
                                             TypeAndOrName &class_type_or_name, 
                                             Address &address)
{
    return false;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
lldb_private::LanguageRuntime *
AppleObjCRuntimeV1::CreateInstance (Process *process, lldb::LanguageType language)
{
    // FIXME: This should be a MacOS or iOS process, and we need to look for the OBJC section to make
    // sure we aren't using the V1 runtime.
    if (language == eLanguageTypeObjC)
    {
        ModuleSP objc_module_sp;
        
        if (AppleObjCRuntime::GetObjCVersion (process, objc_module_sp) == eAppleObjC_V1)
            return new AppleObjCRuntimeV1 (process);
        else
            return NULL;
    }
    else
        return NULL;
}

void
AppleObjCRuntimeV1::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);    
}

void
AppleObjCRuntimeV1::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
AppleObjCRuntimeV1::GetPluginName()
{
    return pluginName;
}

const char *
AppleObjCRuntimeV1::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
AppleObjCRuntimeV1::GetPluginVersion()
{
    return 1;
}

void
AppleObjCRuntimeV1::SetExceptionBreakpoints ()
{
    if (!m_process)
        return;
        
    if (!m_objc_exception_bp_sp)
    {
        m_objc_exception_bp_sp = m_process->GetTarget().CreateBreakpoint (NULL,
                                                                          "objc_exception_throw",
                                                                          eFunctionNameTypeBase, 
                                                                          true);
    }
}

struct BufStruct {
    char contents[2048];
};

ClangUtilityFunction *
AppleObjCRuntimeV1::CreateObjectChecker(const char *name)
{
    std::auto_ptr<BufStruct> buf(new BufStruct);
    
    assert(snprintf(&buf->contents[0], sizeof(buf->contents),
                    "struct __objc_class                                                    \n"
                    "{                                                                      \n"
                    "   struct __objc_class *isa;                                           \n"
                    "   struct __objc_class *super_class;                                   \n"
                    "   const char *name;                                                   \n"
                    "   // rest of struct elided because unused                             \n"
                    "};                                                                     \n"
                    "                                                                       \n"
                    "struct __objc_object                                                   \n"
                    "{                                                                      \n"
                    "   struct __objc_class *isa;                                           \n"
                    "};                                                                     \n"
                    "                                                                       \n"
                    "extern \"C\" void                                                      \n"
                    "%s(void *$__lldb_arg_obj)                                              \n"
                    "{                                                                      \n"
                    "   struct __objc_object *obj = (struct __objc_object*)$__lldb_arg_obj; \n"
                    "   (int)strlen(obj->isa->name);                                        \n"
                    "}                                                                      \n",
                    name) < sizeof(buf->contents));

    return new ClangUtilityFunction(buf->contents, name);
}
