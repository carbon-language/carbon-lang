//===-- SymbolVendor.mm -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SymbolVendor.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"

using namespace lldb;
using namespace lldb_private;


//----------------------------------------------------------------------
// FindPlugin
//
// Platforms can register a callback to use when creating symbol
// vendors to allow for complex debug information file setups, and to
// also allow for finding separate debug information files.
//----------------------------------------------------------------------
SymbolVendor*
SymbolVendor::FindPlugin (Module* module)
{
    std::auto_ptr<SymbolVendor> instance_ap;
    //----------------------------------------------------------------------
    // We currently only have one debug symbol parser...
    //----------------------------------------------------------------------
    SymbolVendorCreateInstance create_callback;
    for (uint32_t idx = 0; (create_callback = PluginManager::GetSymbolVendorCreateCallbackAtIndex(idx)) != NULL; ++idx)
    {
        instance_ap.reset(create_callback(module));

        if (instance_ap.get())
        {
            // TODO: make sure this symbol vendor is what we want. We
            // currently are just returning the first one we find, but
            // we may want to call this function only when we have our
            // main executable module and then give all symbol vendor
            // plug-ins a chance to compete for who wins.
            return instance_ap.release();
        }
    }
    // The default implementation just tries to create debug information using the
    // file representation for the module.
    instance_ap.reset(new SymbolVendor(module));
    if (instance_ap.get())
        instance_ap->AddSymbolFileRepresendation(module->GetObjectFile());
    return instance_ap.release();
}

//----------------------------------------------------------------------
// SymbolVendor constructor
//----------------------------------------------------------------------
SymbolVendor::SymbolVendor(Module *module) :
    ModuleChild(module),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_type_list(),
    m_compile_units(),
    m_sym_file_ap()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SymbolVendor::~SymbolVendor()
{
}

//----------------------------------------------------------------------
// Add a represantion given an object file.
//----------------------------------------------------------------------
void
SymbolVendor::AddSymbolFileRepresendation(ObjectFile *obj_file)
{
    Mutex::Locker locker(m_mutex);
    if (obj_file != NULL)
        m_sym_file_ap.reset(SymbolFile::FindPlugin(obj_file));
}

bool
SymbolVendor::SetCompileUnitAtIndex (CompUnitSP& cu, uint32_t idx)
{
    Mutex::Locker locker(m_mutex);
    const uint32_t num_compile_units = GetNumCompileUnits();
    if (idx < num_compile_units)
    {
        // Fire off an assertion if this compile unit already exists for now.
        // The partial parsing should take care of only setting the compile
        // unit once, so if this assertion fails, we need to make sure that
        // we don't have a race condition, or have a second parse of the same
        // compile unit.
        assert(m_compile_units[idx].get() == NULL);
        m_compile_units[idx] = cu;
        return true;
    }
    return false;
}

uint32_t
SymbolVendor::GetNumCompileUnits()
{
    Mutex::Locker locker(m_mutex);
    if (m_compile_units.empty())
    {
        if (m_sym_file_ap.get())
        {
            // Resize our array of compile unit shared pointers -- which will
            // each remain NULL until someone asks for the actual compile unit
            // information. When this happens, the symbol file will be asked
            // to parse this compile unit information.
            m_compile_units.resize(m_sym_file_ap->GetNumCompileUnits());
        }
    }
    return m_compile_units.size();
}

size_t
SymbolVendor::ParseCompileUnitFunctions (const SymbolContext &sc)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->ParseCompileUnitFunctions(sc);
    return 0;
}

bool
SymbolVendor::ParseCompileUnitLineTable (const SymbolContext &sc)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->ParseCompileUnitLineTable(sc);
    return false;
}

bool
SymbolVendor::ParseCompileUnitSupportFiles (const SymbolContext& sc, FileSpecList& support_files)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->ParseCompileUnitSupportFiles(sc, support_files);
    return false;
}

size_t
SymbolVendor::ParseFunctionBlocks (const SymbolContext &sc)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->ParseFunctionBlocks(sc);
    return 0;
}

size_t
SymbolVendor::ParseTypes (const SymbolContext &sc)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->ParseTypes(sc);
    return 0;
}

size_t
SymbolVendor::ParseVariablesForContext (const SymbolContext& sc)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->ParseVariablesForContext(sc);
    return 0;
}

Type*
SymbolVendor::ResolveTypeUID(lldb::user_id_t type_uid)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->ResolveTypeUID(type_uid);
    return NULL;
}


uint32_t
SymbolVendor::ResolveSymbolContext (const Address& so_addr, uint32_t resolve_scope, SymbolContext& sc)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->ResolveSymbolContext(so_addr, resolve_scope, sc);
    return 0;
}

uint32_t
SymbolVendor::ResolveSymbolContext (const FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->ResolveSymbolContext(file_spec, line, check_inlines, resolve_scope, sc_list);
    return 0;
}

uint32_t
SymbolVendor::FindGlobalVariables (const ConstString &name, bool append, uint32_t max_matches, VariableList& variables)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->FindGlobalVariables(name, append, max_matches, variables);
    return 0;
}

uint32_t
SymbolVendor::FindGlobalVariables (const RegularExpression& regex, bool append, uint32_t max_matches, VariableList& variables)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->FindGlobalVariables(regex, append, max_matches, variables);
    return 0;
}

uint32_t
SymbolVendor::FindFunctions(const ConstString &name, uint32_t name_type_mask, bool append, SymbolContextList& sc_list)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->FindFunctions(name, name_type_mask, append, sc_list);
    return 0;
}

uint32_t
SymbolVendor::FindFunctions(const RegularExpression& regex, bool append, SymbolContextList& sc_list)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->FindFunctions(regex, append, sc_list);
    return 0;
}


uint32_t
SymbolVendor::FindTypes (const SymbolContext& sc, const ConstString &name, bool append, uint32_t max_matches, TypeList& types)
{
    Mutex::Locker locker(m_mutex);
    if (m_sym_file_ap.get())
        return m_sym_file_ap->FindTypes(sc, name, append, max_matches, types);
    if (!append)
        types.Clear();
    return 0;
}

ClangNamespaceDecl
SymbolVendor::FindNamespace(const SymbolContext& sc, const ConstString &name)
{
    Mutex::Locker locker(m_mutex);
    ClangNamespaceDecl namespace_decl;
    if (m_sym_file_ap.get())
        namespace_decl = m_sym_file_ap->FindNamespace (sc, name);
    return namespace_decl;
}

void
SymbolVendor::Dump(Stream *s)
{
    Mutex::Locker locker(m_mutex);
    bool show_context = false;

    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    s->PutCString("SymbolVendor");
    if (m_sym_file_ap.get())
    {
        ObjectFile *objfile = m_sym_file_ap->GetObjectFile();
        if (objfile)
        {
            const FileSpec &objfile_file_spec = objfile->GetFileSpec();
            if (objfile_file_spec)
            {
                s->PutCString(" (");
                objfile_file_spec.Dump(s);
                s->PutChar(')');
            }
        }
    }
    s->EOL();
    s->IndentMore();
    m_type_list.Dump(s, show_context);

    CompileUnitConstIter cu_pos, cu_end;
    cu_end = m_compile_units.end();
    for (cu_pos = m_compile_units.begin(); cu_pos != cu_end; ++cu_pos)
    {
        // We currently only dump the compile units that have been parsed
        if (cu_pos->get())
            (*cu_pos)->Dump(s, show_context);
    }

    s->IndentLess();

}

CompUnitSP
SymbolVendor::GetCompileUnitAtIndex(uint32_t idx)
{
    Mutex::Locker locker(m_mutex);
    CompUnitSP cu_sp;
    const uint32_t num_compile_units = GetNumCompileUnits();
    if (idx < num_compile_units)
    {
        cu_sp = m_compile_units[idx];
        if (cu_sp.get() == NULL)
        {
            m_compile_units[idx] = m_sym_file_ap->ParseCompileUnitAtIndex(idx);
            cu_sp = m_compile_units[idx];
        }
    }
    return cu_sp;
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
SymbolVendor::GetPluginName()
{
    return "SymbolVendor";
}

const char *
SymbolVendor::GetShortPluginName()
{
    return "vendor-default";
}

uint32_t
SymbolVendor::GetPluginVersion()
{
    return 1;
}

void
SymbolVendor::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
SymbolVendor::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
SymbolVendor::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}


