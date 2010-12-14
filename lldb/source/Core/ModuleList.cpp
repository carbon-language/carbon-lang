//===-- ModuleList.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ModuleList.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/Symbols.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/VariableList.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ModuleList constructor
//----------------------------------------------------------------------
ModuleList::ModuleList() :
    m_modules(),
    m_modules_mutex (Mutex::eMutexTypeRecursive)
{
}

//----------------------------------------------------------------------
// Copy constructor
//----------------------------------------------------------------------
ModuleList::ModuleList(const ModuleList& rhs) :
    m_modules(rhs.m_modules)
{
}

//----------------------------------------------------------------------
// Assignment operator
//----------------------------------------------------------------------
const ModuleList&
ModuleList::operator= (const ModuleList& rhs)
{
    if (this != &rhs)
    {
        Mutex::Locker locker(m_modules_mutex);
        m_modules = rhs.m_modules;
    }
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ModuleList::~ModuleList()
{
}

void
ModuleList::Append (ModuleSP &module_sp)
{
    Mutex::Locker locker(m_modules_mutex);
    m_modules.push_back(module_sp);
}

bool
ModuleList::AppendIfNeeded (ModuleSP &module_sp)
{
    Mutex::Locker locker(m_modules_mutex);
    collection::iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        if (pos->get() == module_sp.get())
            return false; // Already in the list
    }
    // Only push module_sp on the list if it wasn't already in there.
    m_modules.push_back(module_sp);
    return true;
}

bool
ModuleList::Remove (ModuleSP &module_sp)
{
    Mutex::Locker locker(m_modules_mutex);
    collection::iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        if (pos->get() == module_sp.get())
        {
            m_modules.erase (pos);
            return true;
        }
    }
    return false;
}

size_t
ModuleList::Remove (ModuleList &module_list)
{
    Mutex::Locker locker(m_modules_mutex);
    size_t num_removed = 0;
    collection::iterator pos, end = module_list.m_modules.end();
    for (pos = module_list.m_modules.begin(); pos != end; ++pos)
    {
        if (Remove (*pos))
            ++num_removed;
    }
    return num_removed;
}



void
ModuleList::Clear()
{
    Mutex::Locker locker(m_modules_mutex);
    m_modules.clear();
}

Module*
ModuleList::GetModulePointerAtIndex (uint32_t idx) const
{
    Mutex::Locker locker(m_modules_mutex);
    if (idx < m_modules.size())
        return m_modules[idx].get();
    return NULL;
}

ModuleSP
ModuleList::GetModuleAtIndex(uint32_t idx)
{
    Mutex::Locker locker(m_modules_mutex);
    ModuleSP module_sp;
    if (idx < m_modules.size())
        module_sp = m_modules[idx];
    return module_sp;
}

size_t
ModuleList::FindFunctions (const ConstString &name, uint32_t name_type_mask, bool append, SymbolContextList &sc_list)
{
    if (!append)
        sc_list.Clear();
    
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->FindFunctions (name, name_type_mask, true, sc_list);
    }
    
    return sc_list.GetSize();
}

uint32_t
ModuleList::FindGlobalVariables (const ConstString &name, bool append, uint32_t max_matches, VariableList& variable_list)
{
    size_t initial_size = variable_list.GetSize();
    Mutex::Locker locker(m_modules_mutex);
    collection::iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->FindGlobalVariables (name, append, max_matches, variable_list);
    }
    return variable_list.GetSize() - initial_size;
}


uint32_t
ModuleList::FindGlobalVariables (const RegularExpression& regex, bool append, uint32_t max_matches, VariableList& variable_list)
{
    size_t initial_size = variable_list.GetSize();
    Mutex::Locker locker(m_modules_mutex);
    collection::iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->FindGlobalVariables (regex, append, max_matches, variable_list);
    }
    return variable_list.GetSize() - initial_size;
}


size_t
ModuleList::FindSymbolsWithNameAndType (const ConstString &name, SymbolType symbol_type, SymbolContextList &sc_list)
{
    Mutex::Locker locker(m_modules_mutex);
    sc_list.Clear();
    collection::iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
        (*pos)->FindSymbolsWithNameAndType (name, symbol_type, sc_list);
    return sc_list.GetSize();
}

class ModuleMatches
{
public:
    //--------------------------------------------------------------
    /// Construct with the user ID to look for.
    //--------------------------------------------------------------
    ModuleMatches (const FileSpec *file_spec_ptr,
                   const ArchSpec *arch_ptr,
                   const UUID *uuid_ptr,
                   const ConstString *object_name) :
        m_file_spec_ptr (file_spec_ptr),
        m_arch_ptr (arch_ptr),
        m_uuid_ptr (uuid_ptr),
        m_object_name (object_name)
    {
    }

    //--------------------------------------------------------------
    /// Unary predicate function object callback.
    //--------------------------------------------------------------
    bool
    operator () (const ModuleSP& module_sp) const
    {
        if (m_file_spec_ptr)
        {
            if (!FileSpec::Equal (*m_file_spec_ptr, module_sp->GetFileSpec(), m_file_spec_ptr->GetDirectory()))
                return false;
        }

        if (m_arch_ptr && m_arch_ptr->IsValid())
        {
            if (module_sp->GetArchitecture() != *m_arch_ptr)
                return false;
        }

        if (m_uuid_ptr && m_uuid_ptr->IsValid())
        {
            if (module_sp->GetUUID() != *m_uuid_ptr)
                return false;
        }

        if (m_object_name)
        {
            if (module_sp->GetObjectName() != *m_object_name)
                return false;
        }
        return true;
    }

private:
    //--------------------------------------------------------------
    // Member variables.
    //--------------------------------------------------------------
    const FileSpec *    m_file_spec_ptr;
    const ArchSpec *    m_arch_ptr;
    const UUID *        m_uuid_ptr;
    const ConstString * m_object_name;
};

size_t
ModuleList::FindModules
(
    const FileSpec *file_spec_ptr,
    const ArchSpec *arch_ptr,
    const UUID *uuid_ptr,
    const ConstString *object_name,
    ModuleList& matching_module_list
) const
{
    size_t existing_matches = matching_module_list.GetSize();
    ModuleMatches matcher (file_spec_ptr, arch_ptr, uuid_ptr, object_name);

    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator end = m_modules.end();
    collection::const_iterator pos;

    for (pos = std::find_if (m_modules.begin(), end, matcher);
         pos != end;
         pos = std::find_if (++pos, end, matcher))
    {
        ModuleSP module_sp(*pos);
        matching_module_list.Append(module_sp);
    }
    return matching_module_list.GetSize() - existing_matches;
}

ModuleSP
ModuleList::FindModule (const Module *module_ptr)
{
    ModuleSP module_sp;

    // Scope for "locker"
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::const_iterator pos, end = m_modules.end();

        for (pos = m_modules.begin(); pos != end; ++pos)
        {
            if ((*pos).get() == module_ptr)
            {
                module_sp = (*pos);
                break;
            }
        }
    }
    return module_sp;

}


uint32_t
ModuleList::FindTypes (const SymbolContext& sc, const ConstString &name, bool append, uint32_t max_matches, TypeList& types)
{
    Mutex::Locker locker(m_modules_mutex);
    
    if (!append)
        types.Clear();

    uint32_t total_matches = 0;
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        if (sc.module_sp.get() == NULL || sc.module_sp.get() == (*pos).get())
            total_matches += (*pos)->FindTypes (sc, name, true, max_matches, types);

        if (total_matches >= max_matches)
            break;
    }
    return total_matches;
}


ModuleSP
ModuleList::FindFirstModuleForFileSpec (const FileSpec &file_spec, const ConstString *object_name)
{
    ModuleSP module_sp;
    ModuleMatches matcher (&file_spec, NULL, NULL, NULL);

    // Scope for "locker"
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::const_iterator end = m_modules.end();
        collection::const_iterator pos = m_modules.begin();

        pos = std::find_if (pos, end, matcher);
        if (pos != end)
            module_sp = (*pos);
    }
    return module_sp;

}


size_t
ModuleList::GetSize() const
{
    size_t size = 0;
    {
        Mutex::Locker locker(m_modules_mutex);
        size = m_modules.size();
    }
    return size;
}


void
ModuleList::Dump(Stream *s) const
{
//  s.Printf("%.*p: ", (int)sizeof(void*) * 2, this);
//  s.Indent();
//  s << "ModuleList\n";

    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->Dump(s);
    }
}

void
ModuleList::LogUUIDAndPaths (LogSP &log_sp, const char *prefix_cstr)
{
    if (log_sp)
    {   
        Mutex::Locker locker(m_modules_mutex);
        char uuid_cstr[256];
        collection::const_iterator pos, begin = m_modules.begin(), end = m_modules.end();
        for (pos = begin; pos != end; ++pos)
        {
            Module *module = pos->get();
            module->GetUUID().GetAsCString (uuid_cstr, sizeof(uuid_cstr));
            const FileSpec &module_file_spec = module->GetFileSpec();
            log_sp->Printf ("%s[%u] %s (%s) \"%s/%s\"", 
                            prefix_cstr ? prefix_cstr : "",
                            (uint32_t)std::distance (begin, pos),
                            uuid_cstr,
                            module->GetArchitecture().AsCString(),
                            module_file_spec.GetDirectory().GetCString(),
                            module_file_spec.GetFilename().GetCString());
        }
    }
}

bool
ModuleList::ResolveFileAddress (lldb::addr_t vm_addr, Address& so_addr)
{
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        if ((*pos)->ResolveFileAddress (vm_addr, so_addr))
            return true;
    }

    return false;
}

uint32_t
ModuleList::ResolveSymbolContextForAddress (const Address& so_addr, uint32_t resolve_scope, SymbolContext& sc)
{
    // The address is already section offset so it has a module
    uint32_t resolved_flags = 0;
    Module *module = so_addr.GetModule();
    if (module)
    {
        resolved_flags = module->ResolveSymbolContextForAddress (so_addr,
                                                                 resolve_scope,
                                                                 sc);
    }
    else
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::const_iterator pos, end = m_modules.end();
        for (pos = m_modules.begin(); pos != end; ++pos)
        {
            resolved_flags = (*pos)->ResolveSymbolContextForAddress (so_addr,
                                                                     resolve_scope,
                                                                     sc);
            if (resolved_flags != 0)
                break;
        }
    }

    return resolved_flags;
}

uint32_t
ModuleList::ResolveSymbolContextForFilePath 
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
ModuleList::ResolveSymbolContextsForFileSpec (const FileSpec &file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list)
{
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->ResolveSymbolContextsForFileSpec (file_spec, line, check_inlines, resolve_scope, sc_list);
    }

    return sc_list.GetSize();
}

uint32_t
ModuleList::GetIndexForModule (const Module *module) const
{
    if (module)
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::const_iterator pos;
        collection::const_iterator begin = m_modules.begin();
        collection::const_iterator end = m_modules.end();
        for (pos = begin; pos != end; ++pos)
        {
            if ((*pos).get() == module)
                return std::distance (begin, pos);
        }
    }
    return LLDB_INVALID_INDEX32;
}

static ModuleList &
GetSharedModuleList ()
{
    static ModuleList g_shared_module_list;
    return g_shared_module_list;
}

const lldb::ModuleSP
ModuleList::GetModuleSP (const Module *module_ptr)
{
    lldb::ModuleSP module_sp;
    if (module_ptr)
    {
        ModuleList &shared_module_list = GetSharedModuleList ();
        module_sp = shared_module_list.FindModule (module_ptr);
        assert (module_sp.get());   // This might fire off a few times and we need to make sure it never fires...
    }
    return module_sp;
}

size_t
ModuleList::FindSharedModules 
(
    const FileSpec& in_file_spec,
    const ArchSpec& arch,
    const UUID *uuid_ptr,
    const ConstString *object_name_ptr,
    ModuleList &matching_module_list
)
{
    ModuleList &shared_module_list = GetSharedModuleList ();
    Mutex::Locker locker(shared_module_list.m_modules_mutex);
    return shared_module_list.FindModules (&in_file_spec, &arch, uuid_ptr, object_name_ptr, matching_module_list);
}

Error
ModuleList::GetSharedModule
(
    const FileSpec& in_file_spec,
    const ArchSpec& arch,
    const UUID *uuid_ptr,
    const ConstString *object_name_ptr,
    off_t object_offset,
    ModuleSP &module_sp,
    ModuleSP *old_module_sp_ptr,
    bool *did_create_ptr
)
{
    ModuleList &shared_module_list = GetSharedModuleList ();
    char path[PATH_MAX];
    char uuid_cstr[64];

    Error error;

    module_sp.reset();

    if (did_create_ptr)
        *did_create_ptr = false;
    if (old_module_sp_ptr)
        old_module_sp_ptr->reset();


    // First just try and get the file where it purports to be (path in
    // in_file_spec), then check and uuid.

    if (in_file_spec)
    {
        // Make sure no one else can try and get or create a module while this
        // function is actively working on it by doing an extra lock on the
        // global mutex list.
        ModuleList matching_module_list;
        Mutex::Locker locker(shared_module_list.m_modules_mutex);
        if (shared_module_list.FindModules (&in_file_spec, &arch, uuid_ptr, object_name_ptr, matching_module_list) > 0)
        {
            module_sp = matching_module_list.GetModuleAtIndex(0);

            // If we didn't have a UUID in mind when looking for the object file,
            // then we should make sure the modification time hasn't changed!
            if (uuid_ptr == NULL)
            {
                TimeValue file_spec_mod_time(in_file_spec.GetModificationTime());
                if (file_spec_mod_time.IsValid())
                {
                    if (file_spec_mod_time != module_sp->GetModificationTime())
                    {
                        if (old_module_sp_ptr)
                            *old_module_sp_ptr = module_sp;
                        shared_module_list.Remove (module_sp);
                        module_sp.reset();
                    }
                }
            }
        }

        if (module_sp.get() == NULL)
        {
            module_sp.reset (new Module (in_file_spec, arch, object_name_ptr, object_offset));
            if (module_sp)
            {
                // If we get in here we got the correct arch, now we just need
                // to verify the UUID if one was given
                if (uuid_ptr && *uuid_ptr != module_sp->GetUUID())
                    module_sp.reset();
                else
                {
                    if (did_create_ptr)
                        *did_create_ptr = true;

                    shared_module_list.Append(module_sp);
                    return error;
                }
            }
        }
    }

    // Either the file didn't exist where at the path, or no path was given, so
    // we now have to use more extreme measures to try and find the appropriate
    // module.

    // Fixup the incoming path in case the path points to a valid file, yet
    // the arch or UUID (if one was passed in) don't match.
    FileSpec file_spec = Symbols::LocateExecutableObjectFile (&in_file_spec, arch.IsValid() ? &arch : NULL, uuid_ptr);

    // Don't look for the file if it appears to be the same one we already
    // checked for above...
    if (file_spec != in_file_spec)
    {
        if (!file_spec.Exists())
        {
            file_spec.GetPath(path, sizeof(path));
            if (file_spec.Exists())
            {
                if (uuid_ptr && uuid_ptr->IsValid())
                    uuid_ptr->GetAsCString(uuid_cstr, sizeof (uuid_cstr));
                else
                    uuid_cstr[0] = '\0';


                if (arch.IsValid())
                {
                    if (uuid_cstr[0])
                        error.SetErrorStringWithFormat("'%s' does not contain the %s architecture and UUID %s.\n", path, arch.AsCString(), uuid_cstr[0]);
                    else
                        error.SetErrorStringWithFormat("'%s' does not contain the %s architecture.\n", path, arch.AsCString());
                }
            }
            else
            {
                error.SetErrorStringWithFormat("'%s' does not exist.\n", path);
            }
            return error;
        }


        // Make sure no one else can try and get or create a module while this
        // function is actively working on it by doing an extra lock on the
        // global mutex list.
        Mutex::Locker locker(shared_module_list.m_modules_mutex);
        ModuleList matching_module_list;
        if (shared_module_list.FindModules (&file_spec, &arch, uuid_ptr, object_name_ptr, matching_module_list) > 0)
        {
            module_sp = matching_module_list.GetModuleAtIndex(0);

            // If we didn't have a UUID in mind when looking for the object file,
            // then we should make sure the modification time hasn't changed!
            if (uuid_ptr == NULL)
            {
                TimeValue file_spec_mod_time(file_spec.GetModificationTime());
                if (file_spec_mod_time.IsValid())
                {
                    if (file_spec_mod_time != module_sp->GetModificationTime())
                    {
                        if (old_module_sp_ptr)
                            *old_module_sp_ptr = module_sp;
                        shared_module_list.Remove (module_sp);
                        module_sp.reset();
                    }
                }
            }
        }

        if (module_sp.get() == NULL)
        {
            module_sp.reset (new Module (file_spec, arch, object_name_ptr, object_offset));
            if (module_sp)
            {
                if (did_create_ptr)
                    *did_create_ptr = true;

                shared_module_list.Append(module_sp);
            }
            else
            {
                file_spec.GetPath(path, sizeof(path));

                if (file_spec)
                {
                    if (arch.IsValid())
                        error.SetErrorStringWithFormat("Unable to open %s architecture in '%s'.\n", arch.AsCString(), path);
                    else
                        error.SetErrorStringWithFormat("Unable to open '%s'.\n", path);
                }
                else
                {
                    if (uuid_ptr && uuid_ptr->IsValid())
                        uuid_ptr->GetAsCString(uuid_cstr, sizeof (uuid_cstr));
                    else
                        uuid_cstr[0] = '\0';

                    if (uuid_cstr[0])
                        error.SetErrorStringWithFormat("Cannot locate a module for UUID '%s'.\n", uuid_cstr[0]);
                    else
                        error.SetErrorStringWithFormat("Cannot locate a module.\n", path, arch.AsCString());
                }
            }
        }
    }

    return error;
}

