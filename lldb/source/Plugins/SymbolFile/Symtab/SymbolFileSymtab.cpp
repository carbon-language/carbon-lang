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
    m_objc_class_name_to_index ()
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
            if (symtab->AppendSymbolIndexesWithType(eSymbolTypeSourceFile, m_source_indexes))
            {
                abilities |= CompileUnits;
            }

            if (symtab->AppendSymbolIndexesWithType(eSymbolTypeCode, Symtab::eDebugYes, Symtab::eVisibilityAny, m_func_indexes))
            {
                symtab->SortSymbolIndexesByValue(m_func_indexes, true);
                abilities |= Functions;
            }

            if (symtab->AppendSymbolIndexesWithType(eSymbolTypeCode, Symtab::eDebugNo, Symtab::eVisibilityAny, m_code_indexes))
            {
                symtab->SortSymbolIndexesByValue(m_code_indexes, true);
                abilities |= Labels;
            }

            if (symtab->AppendSymbolIndexesWithType(eSymbolTypeData, m_data_indexes))
            {
                symtab->SortSymbolIndexesByValue(m_data_indexes, true);
                abilities |= GlobalVariables;
            }
            
            lldb_private::Symtab::IndexCollection objc_class_indexes;
            if (symtab->AppendSymbolIndexesWithType (eSymbolTypeObjCClass, objc_class_indexes))
            {
                abilities |= RuntimeTypes;
                symtab->AppendSymbolNamesToMap (objc_class_indexes,
                                                true,
                                                true,
                                                m_objc_class_name_to_index);
                m_objc_class_name_to_index.Sort();
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
SymbolFileSymtab::FindTypes (const lldb_private::SymbolContext& sc, 
                             const lldb_private::ConstString &name, 
                             const ClangNamespaceDecl *namespace_decl, 
                             bool append, 
                             uint32_t max_matches, 
                             lldb_private::TypeList& types)
{
    if (!append)
        types.Clear();
    
    if (!m_objc_class_name_to_index.IsEmpty())
    {
        TypeMap::iterator iter = m_objc_class_types.find(name);
        
        if (iter != m_objc_class_types.end())
        {
            types.Insert(iter->second);
            return 1;
        }
        
        const Symtab::NameToIndexMap::Entry *match = m_objc_class_name_to_index.FindFirstValueForName(name.GetCString());
        
        if (match == NULL)
            return 0;
                    
        const bool isForwardDecl = false;
        const bool isInternal = true;
        
        ClangASTContext &ast = GetClangASTContext();
        
        lldb::clang_type_t objc_object_type = ast.CreateObjCClass (name.AsCString(), 
                                                                   ast.GetTranslationUnitDecl(), 
                                                                   isForwardDecl, 
                                                                   isInternal);

        ast.StartTagDeclarationDefinition (objc_object_type);

        std::string regex_str("^[-+]\\[");     // Make sure it starts with "+[" or "-["
        regex_str.append(name.AsCString());    // Followed by the class name
        regex_str.append("[ \\(]");            // Followed by a space or '(' (for a category)    
        RegularExpression regex(regex_str.c_str());
        
        Symtab::IndexCollection indices;
        
        lldb::clang_type_t unknown_type = ast.GetUnknownAnyType();
        std::vector<lldb::clang_type_t> arg_types;

        if (m_obj_file->GetSymtab()->FindAllSymbolsMatchingRexExAndType (regex, eSymbolTypeCode, Symtab::eDebugNo, Symtab::eVisibilityAny, indices) != 0)
        {
            for (Symtab::IndexCollection::iterator pos = indices.begin(), end = indices.end();
                 pos != end;
                 ++pos)
            {
                Symbol *symbol = m_obj_file->GetSymtab()->SymbolAtIndex(*pos);
                
                if (!symbol)
                    continue;
                
                const char *signature = symbol->GetName().AsCString();
                
                //printf ("%s: adding '%s'\n", name.GetCString(), signature);
                int num_args = CountMethodArgs(signature);
                
                while (arg_types.size() < num_args)
                    arg_types.push_back(unknown_type);
                
                bool is_variadic = false;
                unsigned type_quals = 0;
                
                lldb::clang_type_t method_type = ast.CreateFunctionType (unknown_type, 
                                                                         arg_types.data(), 
                                                                         num_args, 
                                                                         is_variadic, 
                                                                         type_quals);
                
                ast.AddMethodToObjCObjectType (objc_object_type, 
                                               signature, 
                                               method_type, 
                                               eAccessPublic);
            }
        }
        
        ast.CompleteTagDeclarationDefinition (objc_object_type);

        Declaration decl;
        
        lldb::TypeSP type(new Type (indices[0],
                                    this,
                                    name,
                                    0,      // byte_size
                                    NULL,   // SymbolContextScope*
                                    0,      // encoding_uid
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
