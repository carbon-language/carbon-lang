//
//  TypeSystem.cpp
//  lldb
//
//  Created by Ryan Brown on 3/29/15.
//
//

#include "lldb/Symbol/TypeSystem.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/CompilerType.h"

using namespace lldb_private;

TypeSystem::TypeSystem(LLVMCastKind kind) :
    m_kind (kind),
    m_sym_file (nullptr)
{
}

TypeSystem::~TypeSystem()
{
}

lldb::TypeSystemSP
TypeSystem::CreateInstance (lldb::LanguageType language, const lldb_private::ArchSpec &arch)
{
    uint32_t i = 0;
    TypeSystemCreateInstance create_callback;
    while ((create_callback = PluginManager::GetTypeSystemCreateCallbackAtIndex (i++)) != nullptr)
    {
        lldb::TypeSystemSP type_system_sp = create_callback(language, arch);
        if (type_system_sp)
            return type_system_sp;
    }

    return lldb::TypeSystemSP();
}

CompilerType
TypeSystem::GetLValueReferenceType (lldb::opaque_compiler_type_t type)
{
    return CompilerType();
}

CompilerType
TypeSystem::GetRValueReferenceType (lldb::opaque_compiler_type_t type)
{
    return CompilerType();
}

CompilerType
TypeSystem::AddConstModifier (lldb::opaque_compiler_type_t type)
{
    return CompilerType();
}

CompilerType
TypeSystem::AddVolatileModifier (lldb::opaque_compiler_type_t type)
{
    return CompilerType();
}

CompilerType
TypeSystem::AddRestrictModifier (lldb::opaque_compiler_type_t type)
{
    return CompilerType();
}

CompilerType
TypeSystem::CreateTypedef (lldb::opaque_compiler_type_t type, const char *name, const CompilerDeclContext &decl_ctx)
{
    return CompilerType();
}

CompilerType
TypeSystem::GetBuiltinTypeByName (const ConstString &name)
{
    return CompilerType();
}

CompilerType
TypeSystem::GetTypeForFormatters (void* type)
{
    return CompilerType(this, type);
}

LazyBool
TypeSystem::ShouldPrintAsOneLiner (void* type)
{
    return eLazyBoolCalculate;
}
