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

Module::Module(const FileSpec& file_spec, const ArchSpec& arch, const ConstString *object_name, off_t object_offset) :
    m_mutex (Mutex::eMutexTypeRecursive),
    m_mod_time (file_spec.GetModificationTime()),
    m_arch (arch),
    m_uuid (),
    m_file (file_spec),
    m_flags (),
    m_object_name (),
    m_objfile_ap (),
    m_symfile_ap ()
{
    if (object_name)
        m_object_name = *object_name;
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT);
    if (log)
        log->Printf ("%p Module::Module((%s) '%s/%s%s%s%s')",
                     this,
                     m_arch.AsCString(),
                     m_file.GetDirectory().AsCString(""),
                     m_file.GetFilename().AsCString(""),
                     m_object_name.IsEmpty() ? "" : "(",
                     m_object_name.IsEmpty() ? "" : m_object_name.AsCString(""),
                     m_object_name.IsEmpty() ? "" : ")");
}

Module::~Module()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT);
    if (log)
        log->Printf ("%p Module::~Module((%s) '%s/%s%s%s%s')",
                     this,
                     m_arch.AsCString(),
                     m_file.GetDirectory().AsCString(""),
                     m_file.GetFilename().AsCString(""),
                     m_object_name.IsEmpty() ? "" : "(",
                     m_object_name.IsEmpty() ? "" : m_object_name.AsCString(""),
                     m_object_name.IsEmpty() ? "" : ")");
}


ModuleSP
Module::GetSP ()
{
    return ModuleList::GetModuleSP (this);
}

const UUID&
Module::GetUUID()
{
    Mutex::Locker locker (m_mutex);
    if (m_flags.IsClear(flagsParsedUUID))
    {
        ObjectFile * obj_file = GetObjectFile ();

        if (obj_file != NULL)
        {
            obj_file->GetUUID(&m_uuid);
            m_flags.Set(flagsParsedUUID);
        }
    }
    return m_uuid;
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

//CompUnitSP
//Module::FindCompUnit(lldb::user_id_t uid)
//{
//  CompUnitSP cu_sp;
//  SymbolVendor *symbols = GetSymbolVendor ();
//  if (symbols)
//      cu_sp = symbols->FindCompUnit(uid);
//  return cu_sp;
//}

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
Module::ResolveSymbolContextForFilePath (const char *file_path, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list)
{
    FileSpec file_spec(file_path);
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
Module::FindFunctions(const ConstString &name, uint32_t name_type_mask, bool append, SymbolContextList& sc_list)
{
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        return symbols->FindFunctions(name, name_type_mask, append, sc_list);
    return 0;
}

uint32_t
Module::FindFunctions(const RegularExpression& regex, bool append, SymbolContextList& sc_list)
{
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        return symbols->FindFunctions(regex, append, sc_list);
    return 0;
}

uint32_t
Module::FindTypes (const SymbolContext& sc, const ConstString &name, bool append, uint32_t max_matches, TypeList& types)
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
    if (m_flags.IsClear(flagsSearchedForSymVendor) && can_create)
    {
        ObjectFile *obj_file = GetObjectFile ();
        if (obj_file != NULL)
        {
            Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
            m_symfile_ap.reset(SymbolVendor::FindPlugin(this));
            m_flags.Set (flagsSearchedForSymVendor);
        }
    }
    return m_symfile_ap.get();
}

const FileSpec &
Module::GetFileSpec () const
{
    return m_file;
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
Module::Dump(Stream *s)
{
    Mutex::Locker locker (m_mutex);
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
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
    if (m_flags.IsClear(flagsSearchedForObjParser))
    {
        m_flags.Set (flagsSearchedForObjParser);
        Timer scoped_timer(__PRETTY_FUNCTION__,
                           "Module::GetObjectFile () module = %s", GetFileSpec().GetFilename().AsCString(""));
        m_objfile_ap.reset(ObjectFile::FindPlugin(this, &m_file, 0, m_file.GetByteSize()));
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
            return symtab->FindFirstSymbolWithNameAndType (name, symbol_type);
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
            symtab->FindAllSymbolsMatchingRexExAndType (regex, symbol_type, symbol_indexes);
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
Module::SetArchitecture (const ArchSpec &new_arch)
{
    if (m_arch == new_arch)
        return true;
    else if (!m_arch.IsValid())
    {
        m_arch = new_arch;
        return true;
    }
    else
    {
        return false;
    }

}

