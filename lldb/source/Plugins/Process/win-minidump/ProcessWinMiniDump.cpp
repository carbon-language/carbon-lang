//===-- ProcessWinMiniDump.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProcessWinMiniDump.h"

#include "lldb/Host/windows/windows.h"
#include <DbgHelp.h>

#include <assert.h>
#include <stdlib.h>

#include <mutex>

#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/UnixSignals.h"
#include "Plugins/DynamicLoader/Windows-DYLD/DynamicLoaderWindowsDYLD.h"

#include "ThreadWinMiniDump.h"

using namespace lldb_private;

// Encapsulates the private data for ProcessWinMiniDump.
// TODO(amccarth):  Determine if we need a mutex for access.
class ProcessWinMiniDump::Data
{
public:
    Data();
    ~Data();

    FileSpec m_core_file;
    HANDLE m_dump_file;  // handle to the open minidump file
    HANDLE m_mapping;  // handle to the file mapping for the minidump file
    void * m_base_addr;  // base memory address of the minidump
};

ConstString
ProcessWinMiniDump::GetPluginNameStatic()
{
    static ConstString g_name("win-minidump");
    return g_name;
}

const char *
ProcessWinMiniDump::GetPluginDescriptionStatic()
{
    return "Windows minidump plug-in.";
}

void
ProcessWinMiniDump::Terminate()
{
    PluginManager::UnregisterPlugin(ProcessWinMiniDump::CreateInstance);
}


lldb::ProcessSP
ProcessWinMiniDump::CreateInstance(Target &target, Listener &listener, const FileSpec *crash_file)
{
    lldb::ProcessSP process_sp;
    if (crash_file)
    {
       process_sp.reset(new ProcessWinMiniDump(target, listener, *crash_file));
    }
    return process_sp;
}

bool
ProcessWinMiniDump::CanDebug(Target &target, bool plugin_specified_by_name)
{
    // TODO(amccarth):  Eventually, this needs some actual logic.
    return true;
}

ProcessWinMiniDump::ProcessWinMiniDump(Target& target, Listener &listener,
                                       const FileSpec &core_file) :
    Process(target, listener),
    m_data_up(new Data)
{
    m_data_up->m_core_file = core_file;
}

ProcessWinMiniDump::~ProcessWinMiniDump()
{
    Clear();
    // We need to call finalize on the process before destroying ourselves
    // to make sure all of the broadcaster cleanup goes as planned. If we
    // destruct this class, then Process::~Process() might have problems
    // trying to fully destroy the broadcaster.
    Finalize();
}

ConstString
ProcessWinMiniDump::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessWinMiniDump::GetPluginVersion()
{
    return 1;
}


Error
ProcessWinMiniDump::DoLoadCore()
{
    Error error;

    error = MapMiniDumpIntoMemory(m_data_up->m_core_file.GetCString());
    if (error.Fail())
    {
        return error;
    }

    m_target.SetArchitecture(DetermineArchitecture());
    // TODO(amccarth):  Build the module list.
    // TODO(amccarth):  Read the exeception record.

    return error;

}

DynamicLoader *
ProcessWinMiniDump::GetDynamicLoader()
{
    if (m_dyld_ap.get() == NULL)
        m_dyld_ap.reset (DynamicLoader::FindPlugin(this, DynamicLoaderWindowsDYLD::GetPluginNameStatic().GetCString()));
    return m_dyld_ap.get();
}

bool
ProcessWinMiniDump::UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    assert(m_data_up != nullptr);
    assert(m_data_up->m_base_addr != 0);

    MINIDUMP_DIRECTORY *dir = nullptr;
    void *ptr = nullptr;
    ULONG size = 0;
    if (::MiniDumpReadDumpStream(m_data_up->m_base_addr, ThreadListStream, &dir, &ptr, &size))
    {
        assert(dir->StreamType == ThreadListStream);
        assert(size == dir->Location.DataSize);
        assert(ptr == static_cast<void*>(static_cast<char*>(m_data_up->m_base_addr) + dir->Location.Rva));
        auto thread_list_ptr = static_cast<const MINIDUMP_THREAD_LIST *>(ptr);
        const ULONG32 thread_count = thread_list_ptr->NumberOfThreads;
        assert(thread_count < std::numeric_limits<int>::max());
        for (int i = 0; i < thread_count; ++i) {
            std::shared_ptr<ThreadWinMiniDump> thread_sp(new ThreadWinMiniDump(*this, thread_list_ptr->Threads[i].ThreadId));
            new_thread_list.AddThread(thread_sp);
        }
    }

    return new_thread_list.GetSize(false) > 0;
}

void
ProcessWinMiniDump::RefreshStateAfterStop()
{
}

Error
ProcessWinMiniDump::DoDestroy()
{
    return Error();
}

bool
ProcessWinMiniDump::IsAlive()
{
    return true;
}

size_t
ProcessWinMiniDump::ReadMemory(lldb::addr_t addr, void *buf, size_t size, Error &error)
{
    // Don't allow the caching that lldb_private::Process::ReadMemory does
    // since in core files we have it all cached our our core file anyway.
    return DoReadMemory(addr, buf, size, error);
}

size_t
ProcessWinMiniDump::DoReadMemory(lldb::addr_t addr, void *buf, size_t size, Error &error)
{
    // TODO
    return 0;
}

void
ProcessWinMiniDump::Clear()
{
    m_thread_list.Clear();
}

void
ProcessWinMiniDump::Initialize()
{
    static std::once_flag g_once_flag;

    std::call_once(g_once_flag, []()
    {
        PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                      GetPluginDescriptionStatic(),
                                      CreateInstance);
    });
}

lldb::addr_t
ProcessWinMiniDump::GetImageInfoAddress()
{
    Target *target = &GetTarget();
    ObjectFile *obj_file = target->GetExecutableModule()->GetObjectFile();
    Address addr = obj_file->GetImageInfoAddress(target);

    if (addr.IsValid())
        return addr.GetLoadAddress(target);
    return LLDB_INVALID_ADDRESS;
}

ArchSpec
ProcessWinMiniDump::GetArchitecture()
{
    // TODO
    return ArchSpec();
}


ProcessWinMiniDump::Data::Data() :
    m_dump_file(INVALID_HANDLE_VALUE),
    m_mapping(NULL),
    m_base_addr(nullptr)
{
}

ProcessWinMiniDump::Data::~Data()
{
    if (m_base_addr)
    {
        ::UnmapViewOfFile(m_base_addr);
        m_base_addr = nullptr;
    }
    if (m_mapping)
    {
        ::CloseHandle(m_mapping);
        m_mapping = NULL;
    }
    if (m_dump_file != INVALID_HANDLE_VALUE)
    {
        ::CloseHandle(m_dump_file);
        m_dump_file = INVALID_HANDLE_VALUE;
    }
}

Error
ProcessWinMiniDump::MapMiniDumpIntoMemory(const char *file)
{
    Error error;

    m_data_up->m_dump_file = ::CreateFile(file, GENERIC_READ, FILE_SHARE_READ,
                                          NULL, OPEN_EXISTING,
                                          FILE_ATTRIBUTE_NORMAL, NULL);
    if (m_data_up->m_dump_file == INVALID_HANDLE_VALUE)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }

    m_data_up->m_mapping = ::CreateFileMapping(m_data_up->m_dump_file, NULL,
                                               PAGE_READONLY, 0, 0, NULL);
    if (m_data_up->m_mapping == NULL)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }

    m_data_up->m_base_addr = ::MapViewOfFile(m_data_up->m_mapping, FILE_MAP_READ, 0, 0, 0);
    if (m_data_up->m_base_addr == NULL)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }

    return error;
}


ArchSpec
ProcessWinMiniDump::DetermineArchitecture()
{
    assert(m_data_up != nullptr);
    assert(m_data_up->m_base_addr != 0);

    MINIDUMP_DIRECTORY *dir = nullptr;
    void *ptr = nullptr;
    ULONG size = 0;
    if (::MiniDumpReadDumpStream(m_data_up->m_base_addr, SystemInfoStream, &dir, &ptr, &size))
    {
        assert(dir->StreamType == SystemInfoStream);
        assert(size == dir->Location.DataSize);
        assert(ptr == static_cast<void*>(static_cast<char*>(m_data_up->m_base_addr) + dir->Location.Rva));
        auto system_info_ptr = static_cast<const MINIDUMP_SYSTEM_INFO *>(ptr);
        switch (system_info_ptr->ProcessorArchitecture)
        {
        case PROCESSOR_ARCHITECTURE_INTEL:
            return ArchSpec(eArchTypeCOFF, IMAGE_FILE_MACHINE_I386, LLDB_INVALID_CPUTYPE);
        case PROCESSOR_ARCHITECTURE_AMD64:
            return ArchSpec(eArchTypeCOFF, IMAGE_FILE_MACHINE_AMD64, LLDB_INVALID_CPUTYPE);
        default:
            break;
        }
    }

    return ArchSpec();  // invalid or unknown
}
