//===-- ProcessElfCore.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <stdlib.h>

// Other libraries and framework includes
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/DynamicLoader.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/DynamicLoader/POSIX-DYLD/DynamicLoaderPOSIXDYLD.h"

// Project includes
#include "ProcessElfCore.h"
#include "ThreadElfCore.h"

using namespace lldb_private;

ConstString
ProcessElfCore::GetPluginNameStatic()
{
    static ConstString g_name("elf-core");
    return g_name;
}

const char *
ProcessElfCore::GetPluginDescriptionStatic()
{
    return "ELF core dump plug-in.";
}

void
ProcessElfCore::Terminate()
{
    PluginManager::UnregisterPlugin (ProcessElfCore::CreateInstance);
}


lldb::ProcessSP
ProcessElfCore::CreateInstance (Target &target, Listener &listener, const FileSpec *crash_file)
{
    lldb::ProcessSP process_sp;
    if (crash_file) 
        process_sp.reset(new ProcessElfCore (target, listener, *crash_file));
    return process_sp;
}

bool
ProcessElfCore::CanDebug(Target &target, bool plugin_specified_by_name)
{
    // For now we are just making sure the file exists for a given module
    if (!m_core_module_sp && m_core_file.Exists())
    {
        ModuleSpec core_module_spec(m_core_file, target.GetArchitecture());
        Error error (ModuleList::GetSharedModule (core_module_spec, m_core_module_sp, 
                                                  NULL, NULL, NULL));
        if (m_core_module_sp)
        {
            ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();
            if (core_objfile && core_objfile->GetType() == ObjectFile::eTypeCoreFile)
                return true;
        }
    }
    return false;
}

//----------------------------------------------------------------------
// ProcessElfCore constructor
//----------------------------------------------------------------------
ProcessElfCore::ProcessElfCore(Target& target, Listener &listener,
                               const FileSpec &core_file) :
    Process (target, listener),
    m_core_module_sp (),
    m_core_file (core_file),
    m_dyld_plugin_name (),
    m_thread_data_valid(false),
    m_thread_data(),
    m_core_aranges ()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ProcessElfCore::~ProcessElfCore()
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
ConstString
ProcessElfCore::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessElfCore::GetPluginVersion()
{
    return 1;
}

lldb::addr_t
ProcessElfCore::AddAddressRangeFromLoadSegment(const elf::ELFProgramHeader *header)
{
    lldb::addr_t addr = header->p_vaddr;
    FileRange file_range (header->p_offset, header->p_filesz);
    VMRangeToFileOffset::Entry range_entry(addr, header->p_memsz, file_range);

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

    return addr;
}

//----------------------------------------------------------------------
// Process Control
//----------------------------------------------------------------------
Error
ProcessElfCore::DoLoadCore ()
{
    Error error;
    if (!m_core_module_sp)
    {
        error.SetErrorString ("invalid core module");   
        return error;
    }

    ObjectFileELF *core = (ObjectFileELF *)(m_core_module_sp->GetObjectFile());
    if (core == NULL)
    {
        error.SetErrorString ("invalid core object file");   
        return error;
    }

    const uint32_t num_segments = core->GetProgramHeaderCount();
    if (num_segments == 0)
    {
        error.SetErrorString ("core file has no sections");   
        return error;
    }

    SetCanJIT(false);

    m_thread_data_valid = true;

    bool ranges_are_sorted = true;
    lldb::addr_t vm_addr = 0;
    /// Walk through segments and Thread and Address Map information.
    /// PT_NOTE - Contains Thread and Register information
    /// PT_LOAD - Contains a contiguous range of Process Address Space
    for(uint32_t i = 1; i <= num_segments; i++)
    {
        const elf::ELFProgramHeader *header = core->GetProgramHeaderByIndex(i);
        assert(header != NULL);

        DataExtractor data = core->GetSegmentDataByIndex(i);

        // Parse thread contexts and auxv structure
        if (header->p_type == llvm::ELF::PT_NOTE)
            ParseThreadContextsFromNoteSegment(header, data);

        // PT_LOAD segments contains address map
        if (header->p_type == llvm::ELF::PT_LOAD)
        {
            lldb::addr_t last_addr = AddAddressRangeFromLoadSegment(header);
            if (vm_addr > last_addr)
                ranges_are_sorted = false;
            vm_addr = last_addr;
        }
    }

    if (!ranges_are_sorted)
        m_core_aranges.Sort();

    // Even if the architecture is set in the target, we need to override
    // it to match the core file which is always single arch.
    ArchSpec arch (m_core_module_sp->GetArchitecture());
    switch (arch.GetCore())
    {
        case ArchSpec::eCore_x86_32_i486:
            arch.SetTriple ("i386", m_target.GetPlatform().get());
            break;
        case ArchSpec::eCore_x86_64_x86_64:
            arch.SetTriple ("x86_64-linux-gnu", m_target.GetPlatform().get());
            break;
        default:
            assert(false && "Unhandled core type");
    }
    if (arch.IsValid())
        m_target.SetArchitecture(arch);            

    return error;
}

lldb_private::DynamicLoader *
ProcessElfCore::GetDynamicLoader ()
{
    if (m_dyld_ap.get() == NULL)
        m_dyld_ap.reset (DynamicLoader::FindPlugin(this, DynamicLoaderPOSIXDYLD::GetPluginNameStatic().GetCString()));
    return m_dyld_ap.get();
}

bool
ProcessElfCore::UpdateThreadList (ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    const uint32_t num_threads = GetNumThreadContexts ();
    if (!m_thread_data_valid)
        return false;

    for (lldb::tid_t tid = 0; tid < num_threads; ++tid)
    {
        const ThreadData &td = m_thread_data[tid];
        lldb::ThreadSP thread_sp(new ThreadElfCore (*this, tid, td.prstatus,
                                                    td.prpsinfo, td.fpregset));
        new_thread_list.AddThread (thread_sp);
    }
    return new_thread_list.GetSize(false) > 0;
}

void
ProcessElfCore::RefreshStateAfterStop ()
{
}

Error
ProcessElfCore::DoDestroy ()
{
    return Error();
}

//------------------------------------------------------------------
// Process Queries
//------------------------------------------------------------------

bool
ProcessElfCore::IsAlive ()
{
    return true;
}

//------------------------------------------------------------------
// Process Memory
//------------------------------------------------------------------
size_t
ProcessElfCore::ReadMemory (lldb::addr_t addr, void *buf, size_t size, Error &error)
{
    // Don't allow the caching that lldb_private::Process::ReadMemory does
    // since in core files we have it all cached our our core file anyway.
    return DoReadMemory (addr, buf, size, error);
}

size_t
ProcessElfCore::DoReadMemory (lldb::addr_t addr, void *buf, size_t size, Error &error)
{
    ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();

    if (core_objfile == NULL)
        return 0;

    // Get the address range
    const VMRangeToFileOffset::Entry *address_range = m_core_aranges.FindEntryThatContains (addr);
    if (address_range == NULL || address_range->GetRangeEnd() < addr)
    {
        error.SetErrorStringWithFormat ("core file does not contain 0x%" PRIx64, addr);
        return 0;
    }

    // Convert the address into core file offset
    const lldb::addr_t offset = addr - address_range->GetRangeBase();
    const lldb::addr_t file_start = address_range->data.GetRangeBase();
    const lldb::addr_t file_end = address_range->data.GetRangeEnd();
    size_t bytes_to_read = size; // Number of bytes to read from the core file
    size_t bytes_copied = 0;     // Number of bytes actually read from the core file
    size_t zero_fill_size = 0;   // Padding
    lldb::addr_t bytes_left = 0; // Number of bytes available in the core file from the given address

    if (file_end > offset)
        bytes_left = file_end - offset;

    if (bytes_to_read > bytes_left)
    {
        zero_fill_size = bytes_to_read - bytes_left;
        bytes_to_read = bytes_left;
    }

    // If there is data available on the core file read it
    if (bytes_to_read)
        bytes_copied = core_objfile->CopyData(offset + file_start, bytes_to_read, buf);

    assert(zero_fill_size <= size);
    // Pad remaining bytes
    if (zero_fill_size)
        memset(((char *)buf) + bytes_copied, 0, zero_fill_size);

    return bytes_copied + zero_fill_size;
}

void
ProcessElfCore::Clear()
{
    m_thread_list.Clear();
}

void
ProcessElfCore::Initialize()
{
    static bool g_initialized = false;
    
    if (g_initialized == false)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin (GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance);
    }
}

lldb::addr_t
ProcessElfCore::GetImageInfoAddress()
{
    Target *target = &GetTarget();
    ObjectFile *obj_file = target->GetExecutableModule()->GetObjectFile();
    Address addr = obj_file->GetImageInfoAddress();

    if (addr.IsValid()) 
        return addr.GetLoadAddress(target);
    return LLDB_INVALID_ADDRESS;
}

/// Core files PT_NOTE segment descriptor types
enum {
    NT_PRSTATUS     = 1,
    NT_FPREGSET,
    NT_PRPSINFO,
    NT_TASKSTRUCT,
    NT_PLATFORM,
    NT_AUXV
};

/// Note Structure found in ELF core dumps.
/// This is PT_NOTE type program/segments in the core file.
struct ELFNote
{
    elf::elf_word n_namesz;
    elf::elf_word n_descsz;
    elf::elf_word n_type;

    ELFNote()
    {
        memset(this, 0, sizeof(ELFNote));
    }

    /// Parse an ELFNote entry from the given DataExtractor starting at position
    /// \p offset.
    ///
    /// @param[in] data
    ///    The DataExtractor to read from.
    ///
    /// @param[in,out] offset
    ///    Pointer to an offset in the data.  On return the offset will be
    ///    advanced by the number of bytes read.
    ///
    /// @return
    ///    True if the ELFRel entry was successfully read and false otherwise.
    bool
    Parse(const DataExtractor &data, lldb::offset_t *offset)
    {
        // Read all fields.
        if (!data.GetU32(offset, &n_namesz, 3))
            return false;

        return true;
    }
};

/// Align the given value to next boundary specified by the alignment bytes
static uint32_t
AlignToNext(uint32_t value, int alignment_bytes)
{
    return (value + alignment_bytes - 1) & ~(alignment_bytes - 1);
}

/// Parse Thread context from PT_NOTE segment and store it in the thread list
/// Notes:
/// 1) A PT_NOTE segment is composed of one or more NOTE entries.
/// 2) NOTE Entry contains a standard header followed by variable size data.
///   (see ELFNote structure)
/// 3) A Thread Context in a core file usually described by 3 NOTE entries.
///    a) NT_PRSTATUS - Register context
///    b) NT_PRPSINFO - Process info(pid..)
///    c) NT_FPREGSET - Floating point registers
/// 4) The NOTE entries can be in any order
/// 5) If a core file contains multiple thread contexts then there is two data forms
///    a) Each thread context(2 or more NOTE entries) contained in its own segment (PT_NOTE)
///    b) All thread context is stored in a single segment(PT_NOTE).
///        This case is little tricker since while parsing we have to find where the
///        new thread starts. The current implementation marks begining of 
///        new thread when it finds NT_PRSTATUS or NT_PRPSINFO NOTE entry.
void
ProcessElfCore::ParseThreadContextsFromNoteSegment(const elf::ELFProgramHeader *segment_header, 
                                                   DataExtractor segment_data)
{
    assert(segment_header && segment_header->p_type == llvm::ELF::PT_NOTE);

    lldb::offset_t offset = 0;
    ThreadData *thread_data = NULL;

    // Loop through the NOTE entries in the segment
    while (offset < segment_header->p_filesz)
    {
        static unsigned lead_n_type = -1;
        ELFNote note = ELFNote();
        note.Parse(segment_data, &offset);

        if ((lead_n_type == (unsigned)-1) &&
           ((note.n_type == NT_PRSTATUS) || (note.n_type == NT_PRPSINFO)))
            lead_n_type = note.n_type;

        // Begining of new thread
        if (note.n_type == lead_n_type)
        {
            if (thread_data)
            {
                assert(thread_data->prstatus.GetByteSize() > 0);
                // Add the new thread to thread list
                m_thread_data.push_back(*thread_data);
            }
            thread_data = new ThreadData();
        }

        size_t note_start, note_size;
        note_start = offset + AlignToNext(note.n_namesz, 4);
        note_size = AlignToNext(note.n_descsz, 4);

        // Store the NOTE information in the current thread
        DataExtractor note_data (segment_data, note_start, note_size);
        switch (note.n_type)
        {
            case NT_PRSTATUS:
                thread_data->prstatus = note_data;
                break;
            case NT_FPREGSET:
                thread_data->fpregset = note_data;
                break;
            case NT_PRPSINFO:
                thread_data->prpsinfo = note_data;
                break;
            case NT_AUXV:
                m_auxv = DataExtractor(note_data);
                break;
            default:
                break;
        }

        offset += AlignToNext(note.n_namesz, 4) + note_size;
    }
    // Add last entry in the note section
    if (thread_data && thread_data->prstatus.GetByteSize() > 0)
    {
        m_thread_data.push_back(*thread_data);
    }
}

uint32_t
ProcessElfCore::GetNumThreadContexts ()
{
    if (!m_thread_data_valid)
        DoLoadCore();
    return m_thread_data.size();
}

ArchSpec
ProcessElfCore::GetArchitecture()
{
    ObjectFileELF *core_file = (ObjectFileELF *)(m_core_module_sp->GetObjectFile());
    ArchSpec arch;
    core_file->GetArchitecture(arch);
    return arch;
}

const lldb::DataBufferSP
ProcessElfCore::GetAuxvData()
{
    const uint8_t *start = m_auxv.GetDataStart();
    size_t len = m_auxv.GetByteSize();
    lldb::DataBufferSP buffer(new lldb_private::DataBufferHeap(start, len));
    return buffer;
}

