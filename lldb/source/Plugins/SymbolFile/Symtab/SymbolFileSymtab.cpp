//===-- SymbolFileSymtab.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolFileSymtab.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Timer.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Function.h"

using namespace lldb;
using namespace lldb_private;

void
SymbolFileSymtab::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
SymbolFileSymtab::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
SymbolFileSymtab::GetPluginNameStatic()
{
    return "symbol-file.symtab";
}

const char *
SymbolFileSymtab::GetPluginDescriptionStatic()
{
    return "Reads debug symbols from an object file's symbol table.";
}


SymbolFile*
SymbolFileSymtab::CreateInstance (ObjectFile* obj_file)
{
    return new SymbolFileSymtab(obj_file);
}

SymbolFileSymtab::SymbolFileSymtab(ObjectFile* obj_file) :
    SymbolFile(obj_file),
    m_source_indexes(),
    m_func_indexes(),
    m_code_indexes(),
    m_data_indexes(),
    m_addr_indexes(),
    m_has_objc_symbols(eLazyBoolCalculate)
{
}

SymbolFileSymtab::~SymbolFileSymtab()
{
}

ClangASTContext &       
SymbolFileSymtab::GetClangASTContext ()
{    
    ClangASTContext &ast = m_obj_file->GetModule()->GetClangASTContext();
    
    return ast;
}

bool
SymbolFileSymtab::HasObjCSymbols ()
{
    return (m_abilities & RuntimeTypes) != 0;
}

uint32_t
SymbolFileSymtab::CalculateAbilities ()
{
    uint32_t abilities = 0;
    if (m_obj_file)
    {
        const Symtab *symtab = m_obj_file->GetSymtab();
        if (symtab)
        {

            //----------------------------------------------------------------------
            // The snippet of code below will get the indexes the module symbol
            // table entries that are code, data, or function related (debug info),
            // sort them by value (address) and dump the sorted symbols.
            //----------------------------------------------------------------------
            symtab->AppendSymbolIndexesWithType(eSymbolTypeSourceFile, m_source_indexes);
            if (!m_source_indexes.empty())
            {
                abilities |= CompileUnits;
            }
            symtab->AppendSymbolIndexesWithType(eSymbolTypeCode, Symtab::eDebugYes, Symtab::eVisibilityAny, m_func_indexes);
            if (!m_func_indexes.empty())
            {
                symtab->SortSymbolIndexesByValue(m_func_indexes, true);
                abilities |= Functions;
            }

            symtab->AppendSymbolIndexesWithType(eSymbolTypeCode, Symtab::eDebugNo, Symtab::eVisibilityAny, m_code_indexes);
            if (!m_code_indexes.empty())
            {
                symtab->SortSymbolIndexesByValue(m_code_indexes, true);
                abilities |= Labels;
            }

            symtab->AppendSymbolIndexesWithType(eSymbolTypeData, m_data_indexes);

            if (!m_data_indexes.empty())
            {
                symtab->SortSymbolIndexesByValue(m_data_indexes, true);
                abilities |= GlobalVariables;
            }
            
            symtab->AppendSymbolIndexesWithType(eSymbolTypeObjCClass, m_objc_class_indexes);
            
            if (!m_objc_class_indexes.empty())
            {
                symtab->SortSymbolIndexesByValue(m_objc_class_indexes, true);
                abilities |= RuntimeTypes;
            }
        }
    }
    return abilities;
}

uint32_t
SymbolFileSymtab::GetNumCompileUnits()
{
    // If we don't have any source file symbols we will just have one compile unit for
    // the entire object file
    if (m_source_indexes.empty())
        return 0;

    // If we have any source file symbols we will logically orgnize the object symbols
    // using these.
    return m_source_indexes.size();
}

CompUnitSP
SymbolFileSymtab::ParseCompileUnitAtIndex(uint32_t idx)
{
    CompUnitSP cu_sp;

    // If we don't have any source file symbols we will just have one compile unit for
    // the entire object file
//    if (m_source_indexes.empty())
//    {
//        const FileSpec &obj_file_spec = m_obj_file->GetFileSpec();
//        if (obj_file_spec)
//            cu_sp.reset(new CompileUnit(m_obj_file->GetModule(), NULL, obj_file_spec, 0, eLanguageTypeUnknown));
//
//    }
    /* else */ if (idx < m_source_indexes.size())
    {
        const Symbol *cu_symbol = m_obj_file->GetSymtab()->SymbolAtIndex(m_source_indexes[idx]);
        if (cu_symbol)
            cu_sp.reset(new CompileUnit(m_obj_file->GetModule(), NULL, cu_symbol->GetMangled().GetName().AsCString(), 0, eLanguageTypeUnknown));
    }
    return cu_sp;
}

size_t
SymbolFileSymtab::ParseCompileUnitFunctions (const SymbolContext &sc)
{
    size_t num_added = 0;
    // We must at least have a valid compile unit
    assert (sc.comp_unit != NULL);
    const Symtab *symtab = m_obj_file->GetSymtab();
    const Symbol *curr_symbol = NULL;
    const Symbol *next_symbol = NULL;
//  const char *prefix = m_obj_file->SymbolPrefix();
//  if (prefix == NULL)
//      prefix == "";
//
//  const uint32_t prefix_len = strlen(prefix);

    // If we don't have any source file symbols we will just have one compile unit for
    // the entire object file
    if (m_source_indexes.empty())
    {
        // The only time we will have a user ID of zero is when we don't have
        // and source file symbols and we declare one compile unit for the
        // entire object file
        if (!m_func_indexes.empty())
        {

        }

        if (!m_code_indexes.empty())
        {
//          StreamFile s(stdout);
//          symtab->Dump(&s, m_code_indexes);

            uint32_t idx = 0;   // Index into the indexes
            const uint32_t num_indexes = m_code_indexes.size();
            for (idx = 0; idx < num_indexes; ++idx)
            {
                uint32_t symbol_idx = m_code_indexes[idx];
                curr_symbol = symtab->SymbolAtIndex(symbol_idx);
                if (curr_symbol)
                {
                    // Union of all ranges in the function DIE (if the function is discontiguous)
                    AddressRange func_range(curr_symbol->GetValue(), 0);
                    if (func_range.GetBaseAddress().IsSectionOffset())
                    {
                        uint32_t symbol_size = curr_symbol->GetByteSize();
                        if (symbol_size != 0 && !curr_symbol->GetSizeIsSibling())
                            func_range.SetByteSize(symbol_size);
                        else if (idx + 1 < num_indexes)
                        {
                            next_symbol = symtab->SymbolAtIndex(m_code_indexes[idx + 1]);
                            if (next_symbol)
                            {
                                func_range.SetByteSize(next_symbol->GetValue().GetOffset() - curr_symbol->GetValue().GetOffset());
                            }
                        }

                        FunctionSP func_sp(new Function(sc.comp_unit,
                                                            symbol_idx,                 // UserID is the DIE offset
                                                            LLDB_INVALID_UID,           // We don't have any type info for this function
                                                            curr_symbol->GetMangled(),  // Linker/mangled name
                                                            NULL,                       // no return type for a code symbol...
                                                            func_range));               // first address range

                        if (func_sp.get() != NULL)
                        {
                            sc.comp_unit->AddFunction(func_sp);
                            ++num_added;
                        }
                    }
                }
            }

        }
    }
    else
    {
        // We assume we
    }
    return num_added;
}

bool
SymbolFileSymtab::ParseCompileUnitLineTable (const SymbolContext &sc)
{
    return false;
}

bool
SymbolFileSymtab::ParseCompileUnitSupportFiles (const SymbolContext& sc, FileSpecList &support_files)
{
    return false;
}

size_t
SymbolFileSymtab::ParseFunctionBlocks (const SymbolContext &sc)
{
    return 0;
}


size_t
SymbolFileSymtab::ParseTypes (const SymbolContext &sc)
{
    return 0;
}


size_t
SymbolFileSymtab::ParseVariablesForContext (const SymbolContext& sc)
{
    return 0;
}

Type*
SymbolFileSymtab::ResolveTypeUID(lldb::user_id_t type_uid)
{
    return NULL;
}

lldb::clang_type_t
SymbolFileSymtab::ResolveClangOpaqueTypeDefinition (lldb::clang_type_t clang_Type)
{
    return NULL;
}

ClangNamespaceDecl 
SymbolFileSymtab::FindNamespace (const SymbolContext& sc, const ConstString &name, const ClangNamespaceDecl *namespace_decl)
{
    return ClangNamespaceDecl();
}

uint32_t
SymbolFileSymtab::ResolveSymbolContext (const Address& so_addr, uint32_t resolve_scope, SymbolContext& sc)
{
    if (m_obj_file->GetSymtab() == NULL)
        return 0;

    uint32_t resolved_flags = 0;
    if (resolve_scope & eSymbolContextSymbol)
    {
        sc.symbol = m_obj_file->GetSymtab()->FindSymbolContainingFileAddress(so_addr.GetFileAddress());
        if (sc.symbol)
            resolved_flags |= eSymbolContextSymbol;
    }
    return resolved_flags;
}

uint32_t
SymbolFileSymtab::ResolveSymbolContext (const FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list)
{
    return 0;
}

uint32_t
SymbolFileSymtab::FindGlobalVariables(const ConstString &name, const ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, VariableList& variables)
{
    return 0;
}

uint32_t
SymbolFileSymtab::FindGlobalVariables(const RegularExpression& regex, bool append, uint32_t max_matches, VariableList& variables)
{
    return 0;
}

uint32_t
SymbolFileSymtab::FindFunctions(const ConstString &name, const ClangNamespaceDecl *namespace_decl, uint32_t name_type_mask, bool append, SymbolContextList& sc_list)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolFileSymtab::FindFunctions (name = '%s')",
                        name.GetCString());
    // If we ever support finding STABS or COFF debug info symbols, 
    // we will need to add support here. We are not trying to find symbols
    // here, just "lldb_private::Function" objects that come from complete 
    // debug information. Any symbol queries should go through the symbol
    // table itself in the module's object file.
    return 0;
}

uint32_t
SymbolFileSymtab::FindFunctions(const RegularExpression& regex, bool append, SymbolContextList& sc_list)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolFileSymtab::FindFunctions (regex = '%s')",
                        regex.GetText());
    // If we ever support finding STABS or COFF debug info symbols, 
    // we will need to add support here. We are not trying to find symbols
    // here, just "lldb_private::Function" objects that come from complete 
    // debug information. Any symbol queries should go through the symbol
    // table itself in the module's object file.
    return 0;
}

static int CountMethodArgs(const char *method_signature)
{
    int num_args = 0;
    
    for (const char *colon_pos = strchr(method_signature, ':');
         colon_pos != NULL;
         colon_pos = strchr(colon_pos + 1, ':'))
    {
        num_args++;
    }
    
    return num_args;
}

uint32_t
SymbolFileSymtab::FindTypes (const lldb_private::SymbolContext& sc, const lldb_private::ConstString &name, const ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, lldb_private::TypeList& types)
{
    if (!append)
        types.Clear();
    
    if (HasObjCSymbols())
    {
        TypeMap::iterator iter = m_objc_class_types.find(name);
        
        if (iter != m_objc_class_types.end())
        {
            types.Insert(iter->second);
            return 1;
        }
                    
        std::vector<uint32_t> indices;
        /*const ConstString &name, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& symbol_indexes*/
        if (m_obj_file->GetSymtab()->FindAllSymbolsWithNameAndType(name, lldb::eSymbolTypeAny, Symtab::eDebugNo, Symtab::eVisibilityAny, m_objc_class_indexes) == 0)
            return 0;
        
        const bool isForwardDecl = false;
        const bool isInternal = true;
        
        ClangASTContext &clang_ast_ctx = GetClangASTContext();
        
        lldb::clang_type_t objc_object_type = clang_ast_ctx.CreateObjCClass(name.AsCString(), clang_ast_ctx.GetTranslationUnitDecl(), isForwardDecl, isInternal);
                
        const char *class_method_prefix = "^\\+\\[";
        const char *instance_method_prefix = "^\\-\\[";
        const char *method_suffix = " [a-zA-Z0-9:]+\\]$";
        
        std::string class_method_regexp_str(class_method_prefix);
        class_method_regexp_str.append(name.AsCString());
        class_method_regexp_str.append(method_suffix);
        
        RegularExpression class_method_regexp(class_method_regexp_str.c_str());
        
        indices.clear();
        
        lldb::clang_type_t unknown_type = clang_ast_ctx.GetUnknownAnyType();
        std::vector<lldb::clang_type_t> arg_types;

        if (m_obj_file->GetSymtab()->FindAllSymbolsMatchingRexExAndType(class_method_regexp, eSymbolTypeCode, Symtab::eDebugNo, Symtab::eVisibilityAny, indices) != 0)
        {
            for (std::vector<uint32_t>::iterator ii = indices.begin(), ie = indices.end();
                 ii != ie;
                 ++ii)
            {
                Symbol *symbol = m_obj_file->GetSymtab()->SymbolAtIndex(*ii);
                
                if (!symbol)
                    continue;
                
                const char *signature = symbol->GetName().AsCString();
                
                int num_args = CountMethodArgs(signature);
                
                while (arg_types.size() < num_args)
                    arg_types.push_back(unknown_type);
                
                bool is_variadic = false;
                unsigned type_quals = 0;
                
                lldb::clang_type_t method_type = clang_ast_ctx.CreateFunctionType(unknown_type, arg_types.data(), num_args, is_variadic, type_quals);
                
                clang_ast_ctx.AddMethodToObjCObjectType(objc_object_type, signature, method_type, eAccessPublic);
            }
        }
        
        std::string instance_method_regexp_str(instance_method_prefix);
        instance_method_regexp_str.append(name.AsCString());
        instance_method_regexp_str.append(method_suffix);
        
        RegularExpression instance_method_regexp(instance_method_regexp_str.c_str());
        
        indices.clear();
        
        if (m_obj_file->GetSymtab()->FindAllSymbolsMatchingRexExAndType(instance_method_regexp, eSymbolTypeCode, Symtab::eDebugNo, Symtab::eVisibilityAny, indices) != 0)
        {
            for (std::vector<uint32_t>::iterator ii = indices.begin(), ie = indices.end();
                 ii != ie;
                 ++ii)
            {
                Symbol *symbol = m_obj_file->GetSymtab()->SymbolAtIndex(*ii);
                
                if (!symbol)
                    continue;
                
                const char *signature = symbol->GetName().AsCString();
                
                int num_args = CountMethodArgs(signature);
                
                while (arg_types.size() < num_args)
                    arg_types.push_back(unknown_type);
                
                bool is_variadic = false;
                unsigned type_quals = 0;
                
                lldb::clang_type_t method_type = clang_ast_ctx.CreateFunctionType(unknown_type, arg_types.data(), num_args, is_variadic, type_quals);
                
                clang_ast_ctx.AddMethodToObjCObjectType(objc_object_type, signature, method_type, eAccessPublic);
            }
        }
        
        Declaration decl;
        
        lldb::TypeSP type(new Type (indices[0],
                                    this,
                                    name,
                                    0 /*byte_size*/,
                                    NULL /*SymbolContextScope*/,
                                    0 /*encoding_uid*/,
                                    Type::eEncodingInvalid,
                                    decl,
                                    objc_object_type,
                                    Type::eResolveStateFull));
        
        m_objc_class_types[name] = type;
        
        types.Insert(type);
        
        return 1;
    }

    return 0;
}
//
//uint32_t
//SymbolFileSymtab::FindTypes(const SymbolContext& sc, const RegularExpression& regex, bool append, uint32_t max_matches, TypeList& types)
//{
//  return 0;
//}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
SymbolFileSymtab::GetPluginName()
{
    return "SymbolFileSymtab";
}

const char *
SymbolFileSymtab::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
SymbolFileSymtab::GetPluginVersion()
{
    return 1;
}
