//===-- DynamicLoader.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"

using namespace lldb;
using namespace lldb_private;

DynamicLoader*
DynamicLoader::FindPlugin (Process *process, const char *plugin_name)
{
    DynamicLoaderCreateInstance create_callback = NULL;
    if (plugin_name)
    {
        ConstString const_plugin_name(plugin_name);
        create_callback  = PluginManager::GetDynamicLoaderCreateCallbackForPluginName (const_plugin_name);
        if (create_callback)
        {
            std::unique_ptr<DynamicLoader> instance_ap(create_callback(process, true));
            if (instance_ap.get())
                return instance_ap.release();
        }
    }
    else
    {
        for (uint32_t idx = 0; (create_callback = PluginManager::GetDynamicLoaderCreateCallbackAtIndex(idx)) != NULL; ++idx)
        {
            std::unique_ptr<DynamicLoader> instance_ap(create_callback(process, false));
            if (instance_ap.get())
                return instance_ap.release();
        }
    }
    return NULL;
}


//----------------------------------------------------------------------
// DynamicLoader constructor
//----------------------------------------------------------------------
DynamicLoader::DynamicLoader(Process *process) :
    m_process (process)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
DynamicLoader::~DynamicLoader()
{
}

//----------------------------------------------------------------------
// Accessosors to the global setting as to whether to stop at image
// (shared library) loading/unloading.
//----------------------------------------------------------------------
bool
DynamicLoader::GetStopWhenImagesChange () const
{
    return m_process->GetStopOnSharedLibraryEvents();
}

void
DynamicLoader::SetStopWhenImagesChange (bool stop)
{
    m_process->SetStopOnSharedLibraryEvents (stop);
}

ModuleSP
DynamicLoader::GetTargetExecutable()
{
    Target &target = m_process->GetTarget();
    ModuleSP executable = target.GetExecutableModule();

    if (executable.get())
    {
        if (executable->GetFileSpec().Exists())
        {
            ModuleSpec module_spec (executable->GetFileSpec(), executable->GetArchitecture());
            ModuleSP module_sp (new Module (module_spec));

            // Check if the executable has changed and set it to the target executable if they differ.
            if (module_sp.get() && module_sp->GetUUID().IsValid() && executable->GetUUID().IsValid())
            {
                if (module_sp->GetUUID() != executable->GetUUID())
                    executable.reset();
            }
            else if (executable->FileHasChanged())
            {
                executable.reset();
            }

            if (!executable.get())
            {
                executable = target.GetSharedModule(module_spec);
                if (executable.get() != target.GetExecutableModulePointer())
                {
                    // Don't load dependent images since we are in dyld where we will know
                    // and find out about all images that are loaded
                    const bool get_dependent_images = false;
                    target.SetExecutableModule(executable, get_dependent_images);
                }
            }
        }
    }
    return executable;
}

void
DynamicLoader::UpdateLoadedSections(ModuleSP module, addr_t link_map_addr, addr_t base_addr)
{
    UpdateLoadedSectionsCommon(module, base_addr);
}

void
DynamicLoader::UpdateLoadedSectionsCommon(ModuleSP module, addr_t base_addr)
{
    bool changed;
    const bool base_addr_is_offset = true;
    module->SetLoadAddress(m_process->GetTarget(), base_addr, base_addr_is_offset, changed);
}

void
DynamicLoader::UnloadSections(const ModuleSP module)
{
    UnloadSectionsCommon(module);
}

void
DynamicLoader::UnloadSectionsCommon(const ModuleSP module)
{
    Target &target = m_process->GetTarget();
    const SectionList *sections = GetSectionListFromModule(module);

    assert(sections && "SectionList missing from unloaded module.");

    const size_t num_sections = sections->GetSize();
    for (size_t i = 0; i < num_sections; ++i)
    {
        SectionSP section_sp (sections->GetSectionAtIndex(i));
        target.SetSectionUnloaded(section_sp);
    }
}


const SectionList *
DynamicLoader::GetSectionListFromModule(const ModuleSP module) const
{
    SectionList *sections = nullptr;
    if (module.get())
    {
        ObjectFile *obj_file = module->GetObjectFile();
        if (obj_file)
        {
            sections = obj_file->GetSectionList();
        }
    }
    return sections;
}

ModuleSP
DynamicLoader::LoadModuleAtAddress(const FileSpec &file, addr_t link_map_addr, addr_t base_addr)
{
    Target &target = m_process->GetTarget();
    ModuleList &modules = target.GetImages();
    ModuleSP module_sp;

    ModuleSpec module_spec (file, target.GetArchitecture());
    if ((module_sp = modules.FindFirstModule (module_spec)))
    {
        UpdateLoadedSections(module_sp, link_map_addr, base_addr);
    }
    else if ((module_sp = target.GetSharedModule(module_spec)))
    {
        UpdateLoadedSections(module_sp, link_map_addr, base_addr);
    }
    else
    {
        // Try to fetch the load address of the file from the process. It can be different from the
        // address reported by the linker in case of a file with fixed load address because the
        // linker reports the bias between the load address specified in the file and the actual
        // load address it loaded the file.
        bool is_loaded;
        lldb::addr_t load_addr;
        Error error = m_process->GetFileLoadAddress(file, is_loaded, load_addr);
        if (error.Fail() || !is_loaded)
            load_addr = base_addr;

        if ((module_sp = m_process->ReadModuleFromMemory(file, load_addr)))
        {
            UpdateLoadedSections(module_sp, link_map_addr, base_addr);
            target.GetImages().AppendIfNeeded(module_sp);
        }
    }

    return module_sp;
}

int64_t
DynamicLoader::ReadUnsignedIntWithSizeInBytes(addr_t addr, int size_in_bytes)
{
    Error error;

    uint64_t value = m_process->ReadUnsignedIntegerFromMemory(addr, size_in_bytes, 0, error);
    if (error.Fail())
        return -1;
    else
        return (int64_t)value;
}

addr_t
DynamicLoader::ReadPointer(addr_t addr)
{
    Error error;
    addr_t value = m_process->ReadPointerFromMemory(addr, error);
    if (error.Fail())
        return LLDB_INVALID_ADDRESS;
    else
        return value;
}
