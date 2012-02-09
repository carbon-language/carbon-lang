//===-- ProcessMachCore.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>
#include <stdlib.h>

// C++ Includes
#include "llvm/Support/MachO.h"

// Other libraries and framework includes
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

// Project includes
#include "ProcessMachCore.h"
#include "ThreadMachCore.h"
#include "StopInfoMachException.h"

using namespace lldb;
using namespace lldb_private;

const char *
ProcessMachCore::GetPluginNameStatic()
{
    return "mach-o-core";
}

const char *
ProcessMachCore::GetPluginDescriptionStatic()
{
    return "Mach-O core file debugging plug-in.";
}

void
ProcessMachCore::Terminate()
{
    PluginManager::UnregisterPlugin (ProcessMachCore::CreateInstance);
}


lldb::ProcessSP
ProcessMachCore::CreateInstance (Target &target, Listener &listener, const FileSpec *crash_file)
{
    lldb::ProcessSP process_sp;
    if (crash_file)
        process_sp.reset(new ProcessMachCore (target, listener, *crash_file));
    return process_sp;
}

bool
ProcessMachCore::CanDebug(Target &target, bool plugin_specified_by_name)
{
    if (plugin_specified_by_name)
        return true;

    // For now we are just making sure the file exists for a given module
    if (!m_core_module_sp && m_core_file.Exists())
    {
        Error error (ModuleList::GetSharedModule(m_core_file, target.GetArchitecture(), NULL, NULL, 0, m_core_module_sp, NULL, NULL));

        if (m_core_module_sp)
        {
            const llvm::Triple &triple_ref = m_core_module_sp->GetArchitecture().GetTriple();
            if (triple_ref.getOS() == llvm::Triple::Darwin && 
                triple_ref.getVendor() == llvm::Triple::Apple)
            {
                ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();
                if (core_objfile && core_objfile->GetType() == ObjectFile::eTypeCoreFile)
                    return true;
            }
        }
    }
    return false;
}

//----------------------------------------------------------------------
// ProcessMachCore constructor
//----------------------------------------------------------------------
ProcessMachCore::ProcessMachCore(Target& target, Listener &listener, const FileSpec &core_file) :
    Process (target, listener),
    m_core_aranges (),
    m_core_module_sp (),
    m_core_file (core_file),
    m_shlib_addr (LLDB_INVALID_ADDRESS)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ProcessMachCore::~ProcessMachCore()
{
    Clear();
    // We need to call finalize on the process before destroying ourselves
    // to make sure all of the broadcaster cleanup goes as planned. If we
    // destruct this class, then Process::~Process() might have problems
    // trying to fully destroy the broadcaster.
    Finalize();
}

//----------------------------------------------------------------------
// PluginInterface
//----------------------------------------------------------------------
const char *
ProcessMachCore::GetPluginName()
{
    return "Process debugging plug-in that loads mach-o core files.";
}

const char *
ProcessMachCore::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessMachCore::GetPluginVersion()
{
    return 1;
}

//----------------------------------------------------------------------
// Process Control
//----------------------------------------------------------------------
Error
ProcessMachCore::DoLoadCore ()
{
    Error error;
    DataExtractor data;
    ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();
    if (core_objfile == NULL)
    {
        error.SetErrorString ("invalid core object file");   
        return error;
    }
    SectionList *section_list = core_objfile->GetSectionList();
    if (section_list == NULL)
    {
        error.SetErrorString ("core file has no sections");   
        return error;
    }
        
    const uint32_t num_sections = section_list->GetNumSections(0);
    if (num_sections == 0)
    {
        error.SetErrorString ("core file has no sections");   
        return error;
    }
    bool ranges_are_sorted = true;
    addr_t vm_addr = 0;
    for (uint32_t i=0; i<num_sections; ++i)
    {
        Section *section = section_list->GetSectionAtIndex (i).get();
        if (section)
        {
            lldb::addr_t section_vm_addr = section->GetFileAddress();

            if (m_shlib_addr == LLDB_INVALID_ADDRESS)
            {
                if (core_objfile->ReadSectionData (section, data))
                {
                    uint32_t offset = 0;
                    llvm::MachO::mach_header header;
                    if (data.GetU32(&offset, &header, sizeof(header)/sizeof(uint32_t)))
                    {
                        
                        if (header.magic == llvm::MachO::HeaderMagic32 ||
                            header.magic == llvm::MachO::HeaderMagic64)
                        {
                            // Check MH_EXECUTABLE to see if we can find the mach image
                            // that contains the shared library list. The dynamic loader 
                            // (dyld) is what contains the list for user applications,
                            // and the mach kernel contains a global that has the list 
                            // of kexts to load
                            switch (header.filetype)
                            {
                            case llvm::MachO::HeaderFileTypeDynamicLinkEditor:
                                // Address of dyld "struct mach_header" in the core file
                                m_shlib_addr = section_vm_addr;
                                break;
                            case llvm::MachO::HeaderFileTypeExecutable:
                                // Check MH_EXECUTABLE file types to see if the dynamic link object flag
                                // is NOT set. If it isn't, then we have a mach_kernel.
                                if ((header.flags & llvm::MachO::HeaderFlagBitIsDynamicLinkObject) == 0)
                                {
                                    // Address of the mach kernel "struct mach_header" in the core file.
                                    m_shlib_addr = section_vm_addr;                                        
                                }
                                break;
                            }
                        }
                    }
                }
            }
            FileRange file_range (section->GetFileOffset(), section->GetFileSize());
            VMRangeToFileOffset::Entry range_entry (section_vm_addr,
                                                    section->GetByteSize(),
                                                    file_range);
            
            if (vm_addr > section_vm_addr)
                ranges_are_sorted = false;
            vm_addr = section->GetFileAddress();
            VMRangeToFileOffset::Entry *last_entry = m_core_aranges.Back();
            if (last_entry &&
                last_entry->GetRangeEnd() == range_entry.GetRangeBase() &&
                last_entry->data.GetRangeEnd() == range_entry.data.GetRangeBase())
            {
                last_entry->SetRangeEnd (range_entry.GetRangeEnd());
                last_entry->data.SetRangeEnd (range_entry.data.GetRangeEnd());
            }
            else
            {
                m_core_aranges.Append(range_entry);
            }
        }
    }
    if (!ranges_are_sorted)
    {
        m_core_aranges.Sort();
    }
    if (!m_target.GetArchitecture().IsValid())
        m_target.SetArchitecture(m_core_module_sp->GetArchitecture());

    return error;
}


uint32_t
ProcessMachCore::UpdateThreadList (ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    if (old_thread_list.GetSize(false) == 0)
    {
        // Make up the thread the first time this is called so we can setup our one and only
        // core thread state.
        ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();

        if (core_objfile)
        {
            const uint32_t num_threads = core_objfile->GetNumThreadContexts ();
            for (lldb::tid_t tid = 0; tid < num_threads; ++tid)
            {
                ThreadSP thread_sp(new ThreadMachCore (*this, tid));
                new_thread_list.AddThread (thread_sp);
            }
        }
    }
    else
    {
        const uint32_t num_threads = old_thread_list.GetSize(false);
        for (uint32_t i=0; i<num_threads; ++i)
            new_thread_list.AddThread (old_thread_list.GetThreadAtIndex (i));
    }
    return new_thread_list.GetSize(false);
}

void
ProcessMachCore::RefreshStateAfterStop ()
{
    // Let all threads recover from stopping and do any clean up based
    // on the previous thread state (if any).
    m_thread_list.RefreshStateAfterStop();
    //SetThreadStopInfo (m_last_stop_packet);
}

Error
ProcessMachCore::DoDestroy ()
{
    return Error();
}

//------------------------------------------------------------------
// Process Queries
//------------------------------------------------------------------

bool
ProcessMachCore::IsAlive ()
{
    return true;
}

//------------------------------------------------------------------
// Process Memory
//------------------------------------------------------------------
size_t
ProcessMachCore::DoReadMemory (addr_t addr, void *buf, size_t size, Error &error)
{
    ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();

    if (core_objfile)
    {
        const VMRangeToFileOffset::Entry *core_memory_entry = m_core_aranges.FindEntryThatContains (addr);
        if (core_memory_entry)
        {
            const addr_t offset = addr - core_memory_entry->GetRangeBase();
            const addr_t bytes_left = core_memory_entry->GetRangeEnd() - addr;
            size_t bytes_to_read = size;
            if (bytes_to_read > bytes_left)
                bytes_to_read = bytes_left;
            return core_objfile->CopyData (core_memory_entry->data.GetRangeBase() + offset, bytes_to_read, buf);
        }
        else
        {
            error.SetErrorStringWithFormat ("core file does not contain 0x%llx", addr);
        }
    }
    return 0;
}

void
ProcessMachCore::Clear()
{
    m_thread_list.Clear();
}

void
ProcessMachCore::Initialize()
{
    static bool g_initialized = false;
    
    if (g_initialized == false)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                       GetPluginDescriptionStatic(),
                                       CreateInstance);        
    }
}

addr_t
ProcessMachCore::GetImageInfoAddress()
{
    return m_shlib_addr;
}


