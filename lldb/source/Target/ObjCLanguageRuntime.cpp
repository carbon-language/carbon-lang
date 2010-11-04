//===-- CPPLanguageRuntime.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/Type.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Expression/ClangUtilityFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ObjCLanguageRuntime::~ObjCLanguageRuntime()
{
}

ObjCLanguageRuntime::ObjCLanguageRuntime (Process *process) :
    LanguageRuntime (process)
{

}

void
ObjCLanguageRuntime::AddToMethodCache (lldb::addr_t class_addr, lldb::addr_t selector, lldb::addr_t impl_addr)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
    if (log)
    {
        log->Printf ("Caching: class 0x%llx selector 0x%llx implementation 0x%llx.", class_addr, selector, impl_addr);
    }
    m_impl_cache.insert (std::pair<ClassAndSel,lldb::addr_t> (ClassAndSel(class_addr, selector), impl_addr));
}

lldb::addr_t
ObjCLanguageRuntime::LookupInMethodCache (lldb::addr_t class_addr, lldb::addr_t selector)
{
    MsgImplMap::iterator pos, end = m_impl_cache.end();
    pos = m_impl_cache.find (ClassAndSel(class_addr, selector));
    if (pos != end)
        return (*pos).second;
    return LLDB_INVALID_ADDRESS;
}

ClangUtilityFunction *
ObjCLanguageRuntime::CreateObjectChecker(const char *name)
{
    char buf[256];
    
    assert(snprintf(&buf[0], sizeof(buf), 
                    "extern \"C\" int gdb_object_getClass(void *);"
                    "extern \"C\" void "
                    "%s(void *$__lldb_arg_obj)"
                    "{"
                    "    void **isa_ptr = (void **)$__lldb_arg_obj;"
                    "    if (!isa_ptr || !gdb_class_getClass(*isa_ptr))"
                    "        abort();"
                    "}", 
                    name) < sizeof(buf));

    return new ClangUtilityFunction(buf, name);
}
