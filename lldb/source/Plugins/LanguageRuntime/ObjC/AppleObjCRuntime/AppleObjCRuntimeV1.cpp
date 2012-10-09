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

BreakpointResolverSP
AppleObjCRuntimeV1::CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp)
{
    BreakpointResolverSP resolver_sp;
    
    if (throw_bp)
        resolver_sp.reset (new BreakpointResolverName (bkpt,
                                                       "objc_exception_throw",
                                                       eFunctionNameTypeBase,
                                                       Breakpoint::Exact,
                                                       eLazyBoolNo));
    // FIXME: don't do catch yet.
    return resolver_sp;
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
                    "%s(void *$__lldb_arg_obj, void *$__lldb_arg_selector)                  \n"
                    "{                                                                      \n"
                    "   struct __objc_object *obj = (struct __objc_object*)$__lldb_arg_obj; \n"
                    "   (int)strlen(obj->isa->name);                                        \n"
                    "}                                                                      \n",
                    name) < sizeof(buf->contents));

    return new ClangUtilityFunction(buf->contents, name);
}

// this code relies on the assumption that an Objective-C object always starts
// with an ISA at offset 0.
ObjCLanguageRuntime::ObjCISA
AppleObjCRuntimeV1::GetISA(ValueObject& valobj)
{
//    if (ClangASTType::GetMinimumLanguage(valobj.GetClangAST(),valobj.GetClangType()) != eLanguageTypeObjC)
//        return 0;
    
    // if we get an invalid VO (which might still happen when playing around
    // with pointers returned by the expression parser, don't consider this
    // a valid ObjC object)
    if (valobj.GetValue().GetContextType() == Value::eContextTypeInvalid)
        return 0;
    
    addr_t isa_pointer = valobj.GetPointerValue();
    
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    
    Process *process = exe_ctx.GetProcessPtr();
    if (process)
    {
        uint8_t pointer_size = process->GetAddressByteSize();
        
        Error error;
        return process->ReadUnsignedIntegerFromMemory (isa_pointer,
                                                       pointer_size,
                                                       0,
                                                       error);
    }
    return 0;
}

AppleObjCRuntimeV1::ClassDescriptorV1::ClassDescriptorV1 (ValueObject &isa_pointer)
{
    ObjCISA ptr_value = isa_pointer.GetValueAsUnsigned(0);

    lldb::ProcessSP process_sp = isa_pointer.GetProcessSP();

    Initialize (ptr_value,process_sp);
}

AppleObjCRuntimeV1::ClassDescriptorV1::ClassDescriptorV1 (ObjCISA isa, lldb::ProcessSP process_sp)
{
    Initialize (isa, process_sp);
}

void
AppleObjCRuntimeV1::ClassDescriptorV1::Initialize (ObjCISA isa, lldb::ProcessSP process_sp)
{
    if (!isa || !process_sp)
    {
        m_valid = false;
        return;
    }
    
    m_valid = true;
    
    Error error;
    
    m_isa = process_sp->ReadPointerFromMemory(isa, error);
    
    if (error.Fail())
    {
        m_valid = false;
        return;
    }
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    
    if (!IsPointerValid(m_isa,ptr_size))
    {
        m_valid = false;
        return;
    }

    m_parent_isa = process_sp->ReadPointerFromMemory(m_isa + ptr_size,error);
    
    if (error.Fail())
    {
        m_valid = false;
        return;
    }
    
    if (!IsPointerValid(m_parent_isa,ptr_size,true))
    {
        m_valid = false;
        return;
    }
    
    lldb::addr_t name_ptr = process_sp->ReadPointerFromMemory(m_isa + 2 * ptr_size,error);
    
    if (error.Fail())
    {
        m_valid = false;
        return;
    }
    
    lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024, 0));
    
    size_t count = process_sp->ReadCStringFromMemory(name_ptr, (char*)buffer_sp->GetBytes(), 1024, error);
    
    if (error.Fail())
    {
        m_valid = false;
        return;
    }
    
    if (count)
        m_name = ConstString((char*)buffer_sp->GetBytes());
    else
        m_name = ConstString();
    
    m_instance_size = process_sp->ReadUnsignedIntegerFromMemory(m_isa + 5 * ptr_size, ptr_size, 0, error);
    
    if (error.Fail())
    {
        m_valid = false;
        return;
    }
    
    m_process_wp = lldb::ProcessWP(process_sp);
}

AppleObjCRuntime::ClassDescriptorSP
AppleObjCRuntimeV1::ClassDescriptorV1::GetSuperclass ()
{
    if (!m_valid)
        return AppleObjCRuntime::ClassDescriptorSP();
    ProcessSP process_sp = m_process_wp.lock();
    if (!process_sp)
        return AppleObjCRuntime::ClassDescriptorSP();
    return ObjCLanguageRuntime::ClassDescriptorSP(new AppleObjCRuntimeV1::ClassDescriptorV1(m_parent_isa,process_sp));
}

ObjCLanguageRuntime::ClassDescriptorSP
AppleObjCRuntimeV1::CreateClassDescriptor (ObjCISA isa)
{
    ClassDescriptorSP objc_class_sp;
    if (isa != 0)
        objc_class_sp.reset (new ClassDescriptorV1(isa,m_process->CalculateProcess()));
    return objc_class_sp;
}
