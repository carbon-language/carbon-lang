//===-- Module.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Timer.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolVendor.h"

using namespace lldb;
using namespace lldb_private;

// Shared pointers to modules track module lifetimes in
// targets and in the global module, but this collection
// will track all module objects that are still alive
typedef std::vector<Module *> ModuleCollection;

static ModuleCollection &
GetModuleCollection()
{
    static ModuleCollection g_module_collection;
    return g_module_collection;
}

Mutex &
Module::GetAllocationModuleCollectionMutex()
{
    static Mutex g_module_collection_mutex(Mutex::eMutexTypeRecursive);
    return g_module_collection_mutex;    
}

size_t
Module::GetNumberAllocatedModules ()
{
    Mutex::Locker locker (GetAllocationModuleCollectionMutex());
    return GetModuleCollection().size();
}

Module *
Module::GetAllocatedModuleAtIndex (size_t idx)
{
    Mutex::Locker locker (GetAllocationModuleCollectionMutex());
    ModuleCollection &modules = GetModuleCollection();
    if (idx < modules.size())
        return modules[idx];
    return NULL;
}


    

Module::Module(const FileSpec& file_spec, const ArchSpec& arch, const ConstString *object_name, off_t object_offset) :
    m_mutex (Mutex::eMutexTypeRecursive),
    m_mod_time (file_spec.GetModificationTime()),
    m_arch (arch),
    m_uuid (),
    m_file (file_spec),
    m_platform_file(),
    m_object_name (),
    m_object_offset (object_offset),
    m_objfile_ap (),
    m_symfile_ap (),
    m_ast (),
    m_did_load_objfile (false),
    m_did_load_symbol_vendor (false),
    m_did_parse_uuid (false),
    m_did_init_ast (false),
    m_is_dynamic_loader_module (false)
{
    // Scope for locker below...
    {
        Mutex::Locker locker (GetAllocationModuleCollectionMutex());
        GetModuleCollection().push_back(this);
    }

    if (object_name)
        m_object_name = *object_name;
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Module::Module((%s) '%s/%s%s%s%s')",
                     this,
                     m_arch.GetArchitectureName(),
                     m_file.GetDirectory().AsCString(""),
                     m_file.GetFilename().AsCString(""),
                     m_object_name.IsEmpty() ? "" : "(",
                     m_object_name.IsEmpty() ? "" : m_object_name.AsCString(""),
                     m_object_name.IsEmpty() ? "" : ")");
}

Module::~Module()
{
    // Scope for locker below...
    {
        Mutex::Locker locker (GetAllocationModuleCollectionMutex());
        ModuleCollection &modules = GetModuleCollection();
        ModuleCollection::iterator end = modules.end();
        ModuleCollection::iterator pos = std::find(modules.begin(), end, this);
        if (pos != end)
            modules.erase(pos);
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Module::~Module((%s) '%s/%s%s%s%s')",
                     this,
                     m_arch.GetArchitectureName(),
                     m_file.GetDirectory().AsCString(""),
                     m_file.GetFilename().AsCString(""),
                     m_object_name.IsEmpty() ? "" : "(",
                     m_object_name.IsEmpty() ? "" : m_object_name.AsCString(""),
                     m_object_name.IsEmpty() ? "" : ")");
    // Release any auto pointers before we start tearing down our member 
    // variables since the object file and symbol files might need to make
    // function calls back into this module object. The ordering is important
    // here because symbol files can require the module object file. So we tear
    // down the symbol file first, then the object file.
    m_symfile_ap.reset();
    m_objfile_ap.reset();
}


ModuleSP
Module::GetSP () const
{
    ModuleSP module_sp(const_cast<Module*>(this));
    return module_sp;
}

const lldb_private::UUID&
Module::GetUUID()
{
    Mutex::Locker locker (m_mutex);
    if (m_did_parse_uuid == false)
    {
        ObjectFile * obj_file = GetObjectFile ();

        if (obj_file != NULL)
        {
            obj_file->GetUUID(&m_uuid);
            m_did_parse_uuid = true;
        }
    }
    return m_uuid;
}

ClangASTContext &
Module::GetClangASTContext ()
{
    Mutex::Locker locker (m_mutex);
    if (m_did_init_ast == false)
    {
        ObjectFile * objfile = GetObjectFile();
        ArchSpec object_arch;
        if (objfile && objfile->GetArchitecture(object_arch))
        {
            m_did_init_ast = true;
            m_ast.SetArchitecture (object_arch);
        }
    }
    return m_ast;
}

void
Module::ParseAllDebugSymbols()
{
    Mutex::Locker locker (m_mutex);
    uint32_t num_comp_units = GetNumCompileUnits();
    if (num_comp_units == 0)
        return;

    TargetSP null_target;
    SymbolContext sc(null_target, GetSP());
    uint32_t cu_idx;
    SymbolVendor *symbols = GetSymbolVendor ();

    for (cu_idx = 0; cu_idx < num_comp_units; cu_idx++)
    {
        sc.comp_unit = symbols->GetCompileUnitAtIndex(cu_idx).get();
        if (sc.comp_unit)
        {
            sc.function = NULL;
            symbols->ParseVariablesForContext(sc);

            symbols->ParseCompileUnitFunctions(sc);

            uint32_t func_idx;
            for (func_idx = 0; (sc.function = sc.comp_unit->GetFunctionAtIndex(func_idx).get()) != NULL; ++func_idx)
            {
                symbols->ParseFunctionBlocks(sc);

                // Parse the variables for this function and all its blocks
                symbols->ParseVariablesForContext(sc);
            }


            // Parse all types for this compile unit
            sc.function = NULL;
            symbols->ParseTypes(sc);
        }
    }
}

void
Module::CalculateSymbolContext(SymbolContext* sc)
{
    sc->module_sp = GetSP();
}

Module *
Module::CalculateSymbolContextModule ()
{
    return this;
}

void
Module::DumpSymbolContext(Stream *s)
{
    s->Printf(", Module{0x%8.8x}", this);
}

uint32_t
Module::GetNumCompileUnits()
{
    Mutex::Locker locker (m_mutex);
    Timer scoped_timer(__PRETTY_FUNCTION__, "Module::GetNumCompileUnits (module = %p)", this);
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        return symbols->GetNumCompileUnits();
    return 0;
}

CompUnitSP
Module::GetCompileUnitAtIndex (uint32_t index)
{
    Mutex::Locker locker (m_mutex);
    uint32_t num_comp_units = GetNumCompileUnits ();
    CompUnitSP cu_sp;

    if (index < num_comp_units)
    {
        SymbolVendor *symbols = GetSymbolVendor ();
        if (symbols)
            cu_sp = symbols->GetCompileUnitAtIndex(index);
    }
    return cu_sp;
}

bool
Module::ResolveFileAddress (lldb::addr_t vm_addr, Address& so_addr)
{
    Mutex::Locker locker (m_mutex);
    Timer scoped_timer(__PRETTY_FUNCTION__, "Module::ResolveFileAddress (vm_addr = 0x%llx)", vm_addr);
    ObjectFile* ofile = GetObjectFile();
    if (ofile)
        return so_addr.ResolveAddressUsingFileSections(vm_addr, ofile->GetSectionList());
    return false;
}

uint32_t
Module::ResolveSymbolContextForAddress (const Address& so_addr, uint32_t resolve_scope, SymbolContext& sc)
{
    Mutex::Locker locker (m_mutex);
    uint32_t resolved_flags = 0;

    // Clear the result symbol context in case we don't find anything
    sc.Clear();

    // Get the section from the section/offset address.
    const Section *section = so_addr.GetSection();

    // Make sure the section matches this module before we try and match anything
    if (section && section->GetModule() == this)
    {
        // If the section offset based address resolved itself, then this
        // is the right module.
        sc.module_sp = GetSP();
        resolved_flags |= eSymbolContextModule;

        // Resolve the compile unit, function, block, line table or line
        // entry if requested.
        if (resolve_scope & eSymbolContextCompUnit    ||
            resolve_scope & eSymbolContextFunction    ||
            resolve_scope & eSymbolContextBlock       ||
            resolve_scope & eSymbolContextLineEntry   )
        {
            SymbolVendor *symbols = GetSymbolVendor ();
            if (symbols)
                resolved_flags |= symbols->ResolveSymbolContext (so_addr, resolve_scope, sc);
        }

        // Resolve the symbol if requested, but don't re-look it up if we've already found it.
        if (resolve_scope & eSymbolContextSymbol && !(resolved_flags & eSymbolContextSymbol))
        {
            ObjectFile* ofile = GetObjectFile();
            if (ofile)
            {
                Symtab *symtab = ofile->GetSymtab();
                if (symtab)
                {
                    if (so_addr.IsSectionOffset())
                    {
                        sc.symbol = symtab->FindSymbolContainingFileAddress(so_addr.GetFileAddress());
                        if (sc.symbol)
                            resolved_flags |= eSymbolContextSymbol;
                    }
                }
            }
        }
    }
    return resolved_flags;
}

uint32_t
Module::ResolveSymbolContextForFilePath 
(
    const char *file_path, 
    uint32_t line, 
    bool check_inlines, 
    uint32_t resolve_scope, 
    SymbolContextList& sc_list
)
{
    FileSpec file_spec(file_path, false);
    return ResolveSymbolContextsForFileSpec (file_spec, line, check_inlines, resolve_scope, sc_list);
}

uint32_t
Module::ResolveSymbolContextsForFileSpec (const FileSpec &file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list)
{
    Mutex::Locker locker (m_mutex);
    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "Module::ResolveSymbolContextForFilePath (%s%s%s:%u, check_inlines = %s, resolve_scope = 0x%8.8x)",
                       file_spec.GetDirectory().AsCString(""),
                       file_spec.GetDirectory() ? "/" : "",
                       file_spec.GetFilename().AsCString(""),
                       line,
                       check_inlines ? "yes" : "no",
                       resolve_scope);

    const uint32_t initial_count = sc_list.GetSize();

    SymbolVendor *symbols = GetSymbolVendor  ();
    if (symbols)
        symbols->ResolveSymbolContext (file_spec, line, check_inlines, resolve_scope, sc_list);

    return sc_list.GetSize() - initial_count;
}


uint32_t
Module::FindGlobalVariables(const ConstString &name, bool append, uint32_t max_matches, VariableList& variables)
{
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        return symbols->FindGlobalVariables(name, append, max_matches, variables);
    return 0;
}
uint32_t
Module::FindGlobalVariables(const RegularExpression& regex, bool append, uint32_t max_matches, VariableList& variables)
{
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        return symbols->FindGlobalVariables(regex, append, max_matches, variables);
    return 0;
}

uint32_t
Module::FindCompileUnits (const FileSpec &path,
                          bool append,
                          SymbolContextList &sc_list)
{
    if (!append)
        sc_list.Clear();
    
    const uint32_t start_size = sc_list.GetSize();
    const uint32_t num_compile_units = GetNumCompileUnits();
    SymbolContext sc;
    sc.module_sp = GetSP();
    const bool compare_directory = path.GetDirectory();
    for (uint32_t i=0; i<num_compile_units; ++i)
    {
        sc.comp_unit = GetCompileUnitAtIndex(i).get();
        if (FileSpec::Equal (*sc.comp_unit, path, compare_directory))
            sc_list.Append(sc);
    }
    return sc_list.GetSize() - start_size;
}

uint32_t
Module::FindFunctions (const ConstString &name, 
                       uint32_t name_type_mask, 
                       bool include_symbols, 
                       bool append, 
                       SymbolContextList& sc_list)
{
    if (!append)
        sc_list.Clear();

    const uint32_t start_size = sc_list.GetSize();

    // Find all the functions (not symbols, but debug information functions...
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        symbols->FindFunctions(name, name_type_mask, append, sc_list);

    // Now check our symbol table for symbols that are code symbols if requested
    if (include_symbols)
    {
        ObjectFile *objfile = GetObjectFile();
        if (objfile)
        {
            Symtab *symtab = objfile->GetSymtab();
            if (symtab)
            {
                std::vector<uint32_t> symbol_indexes;
                symtab->FindAllSymbolsWithNameAndType (name, eSymbolTypeCode, Symtab::eDebugAny, Symtab::eVisibilityAny, symbol_indexes);
                const uint32_t num_matches = symbol_indexes.size();
                if (num_matches)
                {
                    const bool merge_symbol_into_function = true;
                    SymbolContext sc(this);
                    for (uint32_t i=0; i<num_matches; i++)
                    {
                        sc.symbol = symtab->SymbolAtIndex(symbol_indexes[i]);
                        sc_list.AppendIfUnique (sc, merge_symbol_into_function);
                    }
                }
            }
        }
    }
    return sc_list.GetSize() - start_size;
}

uint32_t
Module::FindFunctions (const RegularExpression& regex, 
                       bool include_symbols, 
                       bool append, 
                       SymbolContextList& sc_list)
{
    if (!append)
        sc_list.Clear();
    
    const uint32_t start_size = sc_list.GetSize();
    
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        symbols->FindFunctions(regex, append, sc_list);
    // Now check our symbol table for symbols that are code symbols if requested
    if (include_symbols)
    {
        ObjectFile *objfile = GetObjectFile();
        if (objfile)
        {
            Symtab *symtab = objfile->GetSymtab();
            if (symtab)
            {
                std::vector<uint32_t> symbol_indexes;
                symtab->AppendSymbolIndexesMatchingRegExAndType (regex, eSymbolTypeCode, Symtab::eDebugAny, Symtab::eVisibilityAny, symbol_indexes);
                const uint32_t num_matches = symbol_indexes.size();
                if (num_matches)
                {
                    const bool merge_symbol_into_function = true;
                    SymbolContext sc(this);
                    for (uint32_t i=0; i<num_matches; i++)
                    {
                        sc.symbol = symtab->SymbolAtIndex(symbol_indexes[i]);
                        sc_list.AppendIfUnique (sc, merge_symbol_into_function);
                    }
                }
            }
        }
    }
    return sc_list.GetSize() - start_size;
}

uint32_t
Module::FindTypes_Impl (const SymbolContext& sc, const ConstString &name, bool append, uint32_t max_matches, TypeList& types)
{
    Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
    if (sc.module_sp.get() == NULL || sc.module_sp.get() == this)
    {
        SymbolVendor *symbols = GetSymbolVendor ();
        if (symbols)
            return symbols->FindTypes(sc, name, append, max_matches, types);
    }
    return 0;
}

// depending on implementation details, type lookup might fail because of
// embedded spurious namespace:: prefixes. this call strips them, paying
// attention to the fact that a type might have namespace'd type names as
// arguments to templates, and those must not be stripped off
static const char*
StripTypeName(const char* name_cstr)
{
    const char* skip_namespace = strstr(name_cstr, "::");
    const char* template_arg_char = strchr(name_cstr, '<');
    while (skip_namespace != NULL)
    {
        if (template_arg_char != NULL &&
            skip_namespace > template_arg_char) // but namespace'd template arguments are still good to go
            break;
        name_cstr = skip_namespace+2;
        skip_namespace = strstr(name_cstr, "::");
    }
    return name_cstr;
}

uint32_t
Module::FindTypes (const SymbolContext& sc, const ConstString &name, bool append, uint32_t max_matches, TypeList& types)
{
    uint32_t retval = FindTypes_Impl(sc, name, append, max_matches, types);
    
    if (retval == 0)
    {
        const char *stripped = StripTypeName(name.GetCString());
        return FindTypes_Impl(sc, ConstString(stripped), append, max_matches, types);
    }
    else
        return retval;
    
}

//uint32_t
//Module::FindTypes(const SymbolContext& sc, const RegularExpression& regex, bool append, uint32_t max_matches, Type::Encoding encoding, const char *udt_name, TypeList& types)
//{
//  Timer scoped_timer(__PRETTY_FUNCTION__);
//  SymbolVendor *symbols = GetSymbolVendor ();
//  if (symbols)
//      return symbols->FindTypes(sc, regex, append, max_matches, encoding, udt_name, types);
//  return 0;
//
//}

SymbolVendor*
Module::GetSymbolVendor (bool can_create)
{
    Mutex::Locker locker (m_mutex);
    if (m_did_load_symbol_vendor == false && can_create)
    {
        ObjectFile *obj_file = GetObjectFile ();
        if (obj_file != NULL)
        {
            Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
            m_symfile_ap.reset(SymbolVendor::FindPlugin(this));
            m_did_load_symbol_vendor = true;
        }
    }
    return m_symfile_ap.get();
}

void
Module::SetFileSpecAndObjectName (const FileSpec &file, const ConstString &object_name)
{
    // Container objects whose paths do not specify a file directly can call
    // this function to correct the file and object names.
    m_file = file;
    m_mod_time = file.GetModificationTime();
    m_object_name = object_name;
}

const ArchSpec&
Module::GetArchitecture () const
{
    return m_arch;
}

void
Module::GetDescription (Stream *s)
{
    Mutex::Locker locker (m_mutex);

    if (m_arch.IsValid())
        s->Printf("(%s) ", m_arch.GetArchitectureName());

    char path[PATH_MAX];
    if (m_file.GetPath(path, sizeof(path)))
        s->PutCString(path);

    const char *object_name = m_object_name.GetCString();
    if (object_name)
        s->Printf("(%s)", object_name);
}

void
Module::Dump(Stream *s)
{
    Mutex::Locker locker (m_mutex);
    //s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    s->Printf("Module %s/%s%s%s%s\n",
              m_file.GetDirectory().AsCString(),
              m_file.GetFilename().AsCString(),
              m_object_name ? "(" : "",
              m_object_name ? m_object_name.GetCString() : "",
              m_object_name ? ")" : "");

    s->IndentMore();
    ObjectFile *objfile = GetObjectFile ();

    if (objfile)
        objfile->Dump(s);

    SymbolVendor *symbols = GetSymbolVendor ();

    if (symbols)
        symbols->Dump(s);

    s->IndentLess();
}


TypeList*
Module::GetTypeList ()
{
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        return &symbols->GetTypeList();
    return NULL;
}

const ConstString &
Module::GetObjectName() const
{
    return m_object_name;
}

ObjectFile *
Module::GetObjectFile()
{
    Mutex::Locker locker (m_mutex);
    if (m_did_load_objfile == false)
    {
        m_did_load_objfile = true;
        Timer scoped_timer(__PRETTY_FUNCTION__,
                           "Module::GetObjectFile () module = %s", GetFileSpec().GetFilename().AsCString(""));
        m_objfile_ap.reset(ObjectFile::FindPlugin(this, &m_file, m_object_offset, m_file.GetByteSize()));
    }
    return m_objfile_ap.get();
}


const Symbol *
Module::FindFirstSymbolWithNameAndType (const ConstString &name, SymbolType symbol_type)
{
    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "Module::FindFirstSymbolWithNameAndType (name = %s, type = %i)",
                       name.AsCString(),
                       symbol_type);
    ObjectFile *objfile = GetObjectFile();
    if (objfile)
    {
        Symtab *symtab = objfile->GetSymtab();
        if (symtab)
            return symtab->FindFirstSymbolWithNameAndType (name, symbol_type, Symtab::eDebugAny, Symtab::eVisibilityAny);
    }
    return NULL;
}
void
Module::SymbolIndicesToSymbolContextList (Symtab *symtab, std::vector<uint32_t> &symbol_indexes, SymbolContextList &sc_list)
{
    // No need to protect this call using m_mutex all other method calls are
    // already thread safe.

    size_t num_indices = symbol_indexes.size();
    if (num_indices > 0)
    {
        SymbolContext sc;
        CalculateSymbolContext (&sc);
        for (size_t i = 0; i < num_indices; i++)
        {
            sc.symbol = symtab->SymbolAtIndex (symbol_indexes[i]);
            if (sc.symbol)
                sc_list.Append (sc);
        }
    }
}

size_t
Module::FindSymbolsWithNameAndType (const ConstString &name, SymbolType symbol_type, SymbolContextList &sc_list)
{
    // No need to protect this call using m_mutex all other method calls are
    // already thread safe.


    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "Module::FindSymbolsWithNameAndType (name = %s, type = %i)",
                       name.AsCString(),
                       symbol_type);
    const size_t initial_size = sc_list.GetSize();
    ObjectFile *objfile = GetObjectFile ();
    if (objfile)
    {
        Symtab *symtab = objfile->GetSymtab();
        if (symtab)
        {
            std::vector<uint32_t> symbol_indexes;
            symtab->FindAllSymbolsWithNameAndType (name, symbol_type, symbol_indexes);
            SymbolIndicesToSymbolContextList (symtab, symbol_indexes, sc_list);
        }
    }
    return sc_list.GetSize() - initial_size;
}

size_t
Module::FindSymbolsMatchingRegExAndType (const RegularExpression &regex, SymbolType symbol_type, SymbolContextList &sc_list)
{
    // No need to protect this call using m_mutex all other method calls are
    // already thread safe.

    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "Module::FindSymbolsMatchingRegExAndType (regex = %s, type = %i)",
                       regex.GetText(),
                       symbol_type);
    const size_t initial_size = sc_list.GetSize();
    ObjectFile *objfile = GetObjectFile ();
    if (objfile)
    {
        Symtab *symtab = objfile->GetSymtab();
        if (symtab)
        {
            std::vector<uint32_t> symbol_indexes;
            symtab->FindAllSymbolsMatchingRexExAndType (regex, symbol_type, Symtab::eDebugAny, Symtab::eVisibilityAny, symbol_indexes);
            SymbolIndicesToSymbolContextList (symtab, symbol_indexes, sc_list);
        }
    }
    return sc_list.GetSize() - initial_size;
}

const TimeValue &
Module::GetModificationTime () const
{
    return m_mod_time;
}

bool
Module::IsExecutable ()
{
    if (GetObjectFile() == NULL)
        return false;
    else
        return GetObjectFile()->IsExecutable();
}

bool
Module::IsLoadedInTarget (Target *target)
{
    ObjectFile *obj_file = GetObjectFile();
    if (obj_file)
    {
        SectionList *sections = obj_file->GetSectionList();
        if (sections != NULL)
        {
            size_t num_sections = sections->GetSize();
            for (size_t sect_idx = 0; sect_idx < num_sections; sect_idx++)
            {
                SectionSP section_sp = sections->GetSectionAtIndex(sect_idx);
                if (section_sp->GetLoadBaseAddress(target) != LLDB_INVALID_ADDRESS)
                {
                    return true;
                }
            }
        }
    }
    return false;
}
bool 
Module::SetArchitecture (const ArchSpec &new_arch)
{
    if (!m_arch.IsValid())
    {
        m_arch = new_arch;
        return true;
    }    
    return m_arch == new_arch;
}

