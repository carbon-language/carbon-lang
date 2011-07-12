//===-- DynamicLoaderMacOSXKernel.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Target/StackFrame.h"

#include "DynamicLoaderMacOSXKernel.h"

//#define ENABLE_DEBUG_PRINTF // COMMENT THIS LINE OUT PRIOR TO CHECKIN
#ifdef ENABLE_DEBUG_PRINTF
#include <stdio.h>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

using namespace lldb;
using namespace lldb_private;

/// FIXME - The ObjC Runtime trampoline handler doesn't really belong here.
/// I am putting it here so I can invoke it in the Trampoline code here, but
/// it should be moved to the ObjC Runtime support when it is set up.


//----------------------------------------------------------------------
// Create an instance of this class. This function is filled into
// the plugin info class that gets handed out by the plugin factory and
// allows the lldb to instantiate an instance of this class.
//----------------------------------------------------------------------
DynamicLoader *
DynamicLoaderMacOSXKernel::CreateInstance (Process* process, bool force)
{
    bool create = force;
    if (!create)
    {
        Module* exe_module = process->GetTarget().GetExecutableModule().get();
        if (exe_module)
        {
            ObjectFile *object_file = exe_module->GetObjectFile();
            if (object_file)
            {
                SectionList *section_list = object_file->GetSectionList();
                if (section_list)
                {
                    static ConstString g_kld_section_name ("__KLD");
                    if (section_list->FindSectionByName (g_kld_section_name))
                    {
                        create = true;
                    }
                }
            }
        }
        
        if (create)
        {
            const llvm::Triple &triple_ref = process->GetTarget().GetArchitecture().GetTriple();
            create = triple_ref.getOS() == llvm::Triple::Darwin && triple_ref.getVendor() == llvm::Triple::Apple;
        }
    }
    
    if (create)
        return new DynamicLoaderMacOSXKernel (process);
    return NULL;
}

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
DynamicLoaderMacOSXKernel::DynamicLoaderMacOSXKernel (Process* process) :
    DynamicLoader(process),
    m_kernel(),
    m_kext_summary_header_ptr_addr (),
    m_kext_summary_header_addr (),
    m_kext_summary_header (),
    m_break_id (LLDB_INVALID_BREAK_ID),
    m_kext_summaries(),
    m_mutex(Mutex::eMutexTypeRecursive)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
DynamicLoaderMacOSXKernel::~DynamicLoaderMacOSXKernel()
{
    Clear(true);
}

void
DynamicLoaderMacOSXKernel::UpdateIfNeeded()
{
    LoadKernelModuleIfNeeded();
    SetNotificationBreakpointIfNeeded ();
}
//------------------------------------------------------------------
/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
//------------------------------------------------------------------
void
DynamicLoaderMacOSXKernel::DidAttach ()
{
    PrivateInitialize(m_process);
    UpdateIfNeeded();
}

//------------------------------------------------------------------
/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
//------------------------------------------------------------------
void
DynamicLoaderMacOSXKernel::DidLaunch ()
{
    PrivateInitialize(m_process);
    UpdateIfNeeded();
}


//----------------------------------------------------------------------
// Clear out the state of this class.
//----------------------------------------------------------------------
void
DynamicLoaderMacOSXKernel::Clear (bool clear_process)
{
    Mutex::Locker locker(m_mutex);

    if (m_process->IsAlive() && LLDB_BREAK_ID_IS_VALID(m_break_id))
        m_process->ClearBreakpointSiteByID(m_break_id);

    if (clear_process)
        m_process = NULL;
    m_kernel.Clear(false);
    m_kext_summary_header_ptr_addr.Clear();
    m_kext_summary_header_addr.Clear();
    m_kext_summaries.clear();
    m_break_id = LLDB_INVALID_BREAK_ID;
}


//----------------------------------------------------------------------
// Load the kernel module and initialize the "m_kernel" member. Return
// true _only_ if the kernel is loaded the first time through (subsequent
// calls to this function should return false after the kernel has been
// already loaded).
//----------------------------------------------------------------------
void
DynamicLoaderMacOSXKernel::LoadKernelModuleIfNeeded()
{
    if (!m_kext_summary_header_ptr_addr.IsValid())
    {
        m_kernel.Clear(false);
        m_kernel.module_sp = m_process->GetTarget().GetExecutableModule();
        if (m_kernel.module_sp)
        {
            static ConstString mach_header_name ("_mh_execute_header");
            static ConstString kext_summary_symbol ("gLoadedKextSummaries");
            const Symbol *symbol = NULL;
            symbol = m_kernel.module_sp->FindFirstSymbolWithNameAndType (kext_summary_symbol, eSymbolTypeData);
            if (symbol)
                m_kext_summary_header_ptr_addr = symbol->GetValue();

            symbol = m_kernel.module_sp->FindFirstSymbolWithNameAndType (mach_header_name, eSymbolTypeAbsolute);
            if (symbol)
            {
                // The "_mh_execute_header" symbol is absolute and not a section based 
                // symbol that will have a valid address, so we need to resolve it...
                m_process->GetTarget().GetImages().ResolveFileAddress (symbol->GetValue().GetFileAddress(), m_kernel.so_address);
                DataExtractor data; // Load command data
                if (ReadMachHeader (m_kernel, &data))
                {
                    if (m_kernel.header.filetype == llvm::MachO::HeaderFileTypeExecutable)
                    {
                        if (ParseLoadCommands (data, m_kernel))
                            UpdateImageLoadAddress (m_kernel);
                                                
                        // Update all image infos
                        ReadAllKextSummaries ();
                    }
                }
                else
                {
                    m_kernel.Clear(false);
                }
            }
        }
    }
}

bool
DynamicLoaderMacOSXKernel::FindTargetModule (OSKextLoadedKextSummary &image_info, bool can_create, bool *did_create_ptr)
{
    if (did_create_ptr)
        *did_create_ptr = false;
    
    const bool image_info_uuid_is_valid = image_info.uuid.IsValid();

    if (image_info.module_sp)
    {
        if (image_info_uuid_is_valid)
        {
            if (image_info.module_sp->GetUUID() == image_info.uuid)
                return true;
            else
                image_info.module_sp.reset();
        }
        else
            return true;
    }

    ModuleList &target_images = m_process->GetTarget().GetImages();
    if (image_info_uuid_is_valid)
        image_info.module_sp = target_images.FindModule(image_info.uuid);
    
    if (image_info.module_sp)
        return true;
    
    ArchSpec arch (image_info.GetArchitecture ());
    if (can_create)
    {
        if (image_info_uuid_is_valid)
        {
            image_info.module_sp = m_process->GetTarget().GetSharedModule (FileSpec(),
                                                                arch,
                                                                &image_info.uuid);
            if (did_create_ptr)
                *did_create_ptr = image_info.module_sp;
        }
    }
    return image_info.module_sp;
}

bool
DynamicLoaderMacOSXKernel::UpdateCommPageLoadAddress(Module *module)
{
    bool changed = false;
    if (module)
    {
        ObjectFile *image_object_file = module->GetObjectFile();
        if (image_object_file)
        {
            SectionList *section_list = image_object_file->GetSectionList ();
            if (section_list)
            {
                uint32_t num_sections = section_list->GetSize();
                for (uint32_t i=0; i<num_sections; ++i)
                {
                    Section* section = section_list->GetSectionAtIndex (i).get();
                    if (section)
                    {
                        const addr_t new_section_load_addr = section->GetFileAddress ();
                        const addr_t old_section_load_addr = m_process->GetTarget().GetSectionLoadList().GetSectionLoadAddress (section);
                        if (old_section_load_addr == LLDB_INVALID_ADDRESS ||
                            old_section_load_addr != new_section_load_addr)
                        {
                            if (m_process->GetTarget().GetSectionLoadList().SetSectionLoadAddress (section, section->GetFileAddress ()))
                                changed = true;
                        }
                    }
                }
            }
        }
    }
    return changed;
}

//----------------------------------------------------------------------
// Update the load addresses for all segments in MODULE using the
// updated INFO that is passed in.
//----------------------------------------------------------------------
bool
DynamicLoaderMacOSXKernel::UpdateImageLoadAddress (OSKextLoadedKextSummary& info)
{
    Module *module = info.module_sp.get();
    bool changed = false;
    if (module)
    {
        ObjectFile *image_object_file = module->GetObjectFile();
        if (image_object_file)
        {
            SectionList *section_list = image_object_file->GetSectionList ();
            if (section_list)
            {
                // We now know the slide amount, so go through all sections
                // and update the load addresses with the correct values.
                uint32_t num_segments = info.segments.size();
                for (uint32_t i=0; i<num_segments; ++i)
                {
                    SectionSP section_sp(section_list->FindSectionByName(info.segments[i].name));
                    const addr_t new_section_load_addr = info.segments[i].vmaddr;
                    if (section_sp)
                    {
                        const addr_t old_section_load_addr = m_process->GetTarget().GetSectionLoadList().GetSectionLoadAddress (section_sp.get());
                        if (old_section_load_addr == LLDB_INVALID_ADDRESS ||
                            old_section_load_addr != new_section_load_addr)
                        {
                            if (m_process->GetTarget().GetSectionLoadList().SetSectionLoadAddress (section_sp.get(), new_section_load_addr))
                                changed = true;
                        }
                    }
                    else
                    {
                        fprintf (stderr, 
                                 "warning: unable to find and load segment named '%s' at 0x%llx in '%s/%s' in macosx dynamic loader plug-in.\n",
                                 info.segments[i].name.AsCString("<invalid>"),
                                 (uint64_t)new_section_load_addr,
                                 image_object_file->GetFileSpec().GetDirectory().AsCString(),
                                 image_object_file->GetFileSpec().GetFilename().AsCString());
                    }
                }
            }
        }
    }
    return changed;
}

//----------------------------------------------------------------------
// Update the load addresses for all segments in MODULE using the
// updated INFO that is passed in.
//----------------------------------------------------------------------
bool
DynamicLoaderMacOSXKernel::UnloadImageLoadAddress (OSKextLoadedKextSummary& info)
{
    Module *module = info.module_sp.get();
    bool changed = false;
    if (module)
    {
        ObjectFile *image_object_file = module->GetObjectFile();
        if (image_object_file)
        {
            SectionList *section_list = image_object_file->GetSectionList ();
            if (section_list)
            {
                uint32_t num_segments = info.segments.size();
                for (uint32_t i=0; i<num_segments; ++i)
                {
                    SectionSP section_sp(section_list->FindSectionByName(info.segments[i].name));
                    if (section_sp)
                    {
                        const addr_t old_section_load_addr = info.segments[i].vmaddr;
                        if (m_process->GetTarget().GetSectionLoadList().SetSectionUnloaded (section_sp.get(), old_section_load_addr))
                            changed = true;
                    }
                    else
                    {
                        fprintf (stderr, 
                                 "warning: unable to find and unload segment named '%s' in '%s/%s' in macosx dynamic loader plug-in.\n",
                                 info.segments[i].name.AsCString("<invalid>"),
                                 image_object_file->GetFileSpec().GetDirectory().AsCString(),
                                 image_object_file->GetFileSpec().GetFilename().AsCString());
                    }
                }
            }
        }
    }
    return changed;
}


//----------------------------------------------------------------------
// Static callback function that gets called when our DYLD notification
// breakpoint gets hit. We update all of our image infos and then
// let our super class DynamicLoader class decide if we should stop
// or not (based on global preference).
//----------------------------------------------------------------------
bool
DynamicLoaderMacOSXKernel::BreakpointHitCallback (void *baton, 
                                                  StoppointCallbackContext *context, 
                                                  user_id_t break_id, 
                                                  user_id_t break_loc_id)
{    
    return static_cast<DynamicLoaderMacOSXKernel*>(baton)->BreakpointHit (context, break_id, break_loc_id);    
}

bool
DynamicLoaderMacOSXKernel::BreakpointHit (StoppointCallbackContext *context, 
                                          user_id_t break_id, 
                                          user_id_t break_loc_id)
{    
    LogSP log(GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER));
    if (log)
        log->Printf ("DynamicLoaderMacOSXKernel::BreakpointHit (...)\n");

    ReadAllKextSummaries ();
    
    if (log)
        PutToLog(log.get());

    return GetStopWhenImagesChange();
}


bool
DynamicLoaderMacOSXKernel::ReadKextSummaryHeader ()
{
    Mutex::Locker locker(m_mutex);

    // the all image infos is already valid for this process stop ID

    m_kext_summaries.clear();
    if (m_kext_summary_header_ptr_addr.IsValid())
    {
        const uint32_t addr_size = m_kernel.GetAddressByteSize ();
        const ByteOrder byte_order = m_kernel.GetByteOrder();
        Error error;
        // Read enough bytes for a "OSKextLoadedKextSummaryHeader" structure
        // which is currenty 4 uint32_t and a pointer.
        uint8_t buf[24];
        DataExtractor data (buf, sizeof(buf), byte_order, addr_size);
        const size_t count = 4 * sizeof(uint32_t) + addr_size;
        const bool prefer_file_cache = false;
        if (m_process->GetTarget().ReadPointerFromMemory (m_kext_summary_header_ptr_addr, 
                                                          prefer_file_cache,
                                                          error,
                                                          m_kext_summary_header_addr))
        {
            // We got a valid address for our kext summary header and make sure it isn't NULL
            if (m_kext_summary_header_addr.IsValid() && 
                m_kext_summary_header_addr.GetFileAddress() != 0)
            {
                const size_t bytes_read = m_process->GetTarget().ReadMemory (m_kext_summary_header_addr, prefer_file_cache, buf, count, error);
                if (bytes_read == count)
                {
                    uint32_t offset = 0;
                    m_kext_summary_header.version       = data.GetU32(&offset);
                    m_kext_summary_header.entry_size    = data.GetU32(&offset);
                    m_kext_summary_header.entry_count   = data.GetU32(&offset);
                    m_kext_summary_header.reserved      = data.GetU32(&offset);
                    return true;
                }
            }
        }
    }
    m_kext_summary_header_addr.Clear();
    return false;
}


bool
DynamicLoaderMacOSXKernel::ParseKextSummaries (const Address &kext_summary_addr, 
                                               uint32_t count)
{
    OSKextLoadedKextSummary::collection kext_summaries;
    LogSP log(GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER));
    if (log)
        log->Printf ("Adding %d modules.\n");
        
    Mutex::Locker locker(m_mutex);

    if (!ReadKextSummaries (kext_summary_addr, count, kext_summaries))
        return false;

    for (uint32_t i = 0; i < count; i++)
    {
        if (!kext_summaries[i].UUIDValid())
        {
            DataExtractor data; // Load command data
            if (!ReadMachHeader (kext_summaries[i], &data))
                continue;
            
            ParseLoadCommands (data, kext_summaries[i]);
        }
    }
    bool return_value = AddModulesUsingImageInfos (kext_summaries);
    return return_value;
}

// Adds the modules in image_infos to m_kext_summaries.  
// NB don't call this passing in m_kext_summaries.

bool
DynamicLoaderMacOSXKernel::AddModulesUsingImageInfos (OSKextLoadedKextSummary::collection &image_infos)
{
    // Now add these images to the main list.
    ModuleList loaded_module_list;
    
    for (uint32_t idx = 0; idx < image_infos.size(); ++idx)
    {
        m_kext_summaries.push_back(image_infos[idx]);
        
        if (FindTargetModule (image_infos[idx], true, NULL))
        {
            // UpdateImageLoadAddress will return true if any segments
            // change load address. We need to check this so we don't
            // mention that all loaded shared libraries are newly loaded
            // each time we hit out dyld breakpoint since dyld will list all
            // shared libraries each time.
            if (UpdateImageLoadAddress (image_infos[idx]))
            {
                loaded_module_list.AppendIfNeeded (image_infos[idx].module_sp);
            }
        }
    }
    
    if (loaded_module_list.GetSize() > 0)
    {
        // FIXME: This should really be in the Runtime handlers class, which should get
        // called by the target's ModulesDidLoad, but we're doing it all locally for now 
        // to save time.
        // Also, I'm assuming there can be only one libobjc dylib loaded...
        
        ObjCLanguageRuntime *objc_runtime = m_process->GetObjCLanguageRuntime();
        if (objc_runtime != NULL && !objc_runtime->HasReadObjCLibrary())
        {
            size_t num_modules = loaded_module_list.GetSize();
            for (int i = 0; i < num_modules; i++)
            {
                if (objc_runtime->IsModuleObjCLibrary (loaded_module_list.GetModuleAtIndex (i)))
                {
                    objc_runtime->ReadObjCLibrary (loaded_module_list.GetModuleAtIndex (i));
                    break;
                }
            }
        }
//        if (log)
//            loaded_module_list.LogUUIDAndPaths (log, "DynamicLoaderMacOSXKernel::ModulesDidLoad");
        m_process->GetTarget().ModulesDidLoad (loaded_module_list);
    }
    return true;
}


uint32_t
DynamicLoaderMacOSXKernel::ReadKextSummaries (const Address &kext_summary_addr,
                                              uint32_t image_infos_count, 
                                              OSKextLoadedKextSummary::collection &image_infos)
{
    const ByteOrder endian = m_kernel.GetByteOrder();
    const uint32_t addr_size = m_kernel.GetAddressByteSize();

    image_infos.resize(image_infos_count);
    const size_t count = image_infos.size() * m_kext_summary_header.entry_size;
    DataBufferHeap data(count, 0);
    Error error;
    const bool prefer_file_cache = false;
    const size_t bytes_read = m_process->GetTarget().ReadMemory (kext_summary_addr, 
                                                                 prefer_file_cache,
                                                                 data.GetBytes(), 
                                                                 data.GetByteSize(),
                                                                 error);
    if (bytes_read == count)
    {
        uint32_t offset = 0;
        DataExtractor extractor (data.GetBytes(), data.GetByteSize(), endian, addr_size);
        uint32_t i=0;
        for (; i < image_infos.size() && extractor.ValidOffsetForDataOfSize(offset, m_kext_summary_header.entry_size); ++i)
        {
            const void *name_data = extractor.GetData(&offset, KERNEL_MODULE_MAX_NAME);
            if (name_data == NULL)
                break;
            memcpy (image_infos[i].name, name_data, KERNEL_MODULE_MAX_NAME);
            image_infos[i].uuid.SetBytes(extractor.GetData (&offset, 16));
            image_infos[i].address          = extractor.GetU64(&offset);
            if (!image_infos[i].so_address.SetLoadAddress (image_infos[i].address, &m_process->GetTarget()))
                m_process->GetTarget().GetImages().ResolveFileAddress (image_infos[i].address, image_infos[i].so_address);
            image_infos[i].size             = extractor.GetU64(&offset);
            image_infos[i].version          = extractor.GetU64(&offset);
            image_infos[i].load_tag         = extractor.GetU32(&offset);
            image_infos[i].flags            = extractor.GetU32(&offset);
            image_infos[i].reference_list   = extractor.GetU64(&offset);
        }
        if (i < image_infos.size())
            image_infos.resize(i);
    }
    else
    {
        image_infos.clear();
    }
    return image_infos.size();
}

bool
DynamicLoaderMacOSXKernel::ReadAllKextSummaries ()
{
    LogSP log(GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER));
    
    Mutex::Locker locker(m_mutex);
    
    if (ReadKextSummaryHeader ())
    {
        if (m_kext_summary_header.entry_count > 0 && m_kext_summary_header_addr.IsValid())
        {
            Address summary_addr (m_kext_summary_header_addr);
            summary_addr.Slide(16);
            if (!ParseKextSummaries (summary_addr, m_kext_summary_header.entry_count))
            {
                m_kext_summaries.clear();
            }
            return true;
        }
    }
    return false;
}

//----------------------------------------------------------------------
// Read a mach_header at ADDR into HEADER, and also fill in the load
// command data into LOAD_COMMAND_DATA if it is non-NULL.
//
// Returns true if we succeed, false if we fail for any reason.
//----------------------------------------------------------------------
bool
DynamicLoaderMacOSXKernel::ReadMachHeader (OSKextLoadedKextSummary& kext_summary, DataExtractor *load_command_data)
{
    DataBufferHeap header_bytes(sizeof(llvm::MachO::mach_header), 0);
    Error error;
    const bool prefer_file_cache = false;
    size_t bytes_read = m_process->GetTarget().ReadMemory (kext_summary.so_address,
                                                           prefer_file_cache,
                                                           header_bytes.GetBytes(), 
                                                           header_bytes.GetByteSize(), 
                                                           error);
    if (bytes_read == sizeof(llvm::MachO::mach_header))
    {
        uint32_t offset = 0;
        ::memset (&kext_summary.header, 0, sizeof(kext_summary.header));

        // Get the magic byte unswapped so we can figure out what we are dealing with
        DataExtractor data(header_bytes.GetBytes(), header_bytes.GetByteSize(), endian::InlHostByteOrder(), 4);
        kext_summary.header.magic = data.GetU32(&offset);
        Address load_cmd_addr = kext_summary.so_address;
        data.SetByteOrder(DynamicLoaderMacOSXKernel::GetByteOrderFromMagic(kext_summary.header.magic));
        switch (kext_summary.header.magic)
        {
        case llvm::MachO::HeaderMagic32:
        case llvm::MachO::HeaderMagic32Swapped:
            data.SetAddressByteSize(4);
            load_cmd_addr.Slide (sizeof(llvm::MachO::mach_header));
            break;

        case llvm::MachO::HeaderMagic64:
        case llvm::MachO::HeaderMagic64Swapped:
            data.SetAddressByteSize(8);
            load_cmd_addr.Slide (sizeof(llvm::MachO::mach_header_64));
            break;

        default:
            return false;
        }

        // Read the rest of dyld's mach header
        if (data.GetU32(&offset, &kext_summary.header.cputype, (sizeof(llvm::MachO::mach_header)/sizeof(uint32_t)) - 1))
        {
            if (load_command_data == NULL)
                return true; // We were able to read the mach_header and weren't asked to read the load command bytes

            DataBufferSP load_cmd_data_sp(new DataBufferHeap(kext_summary.header.sizeofcmds, 0));

            size_t load_cmd_bytes_read = m_process->GetTarget().ReadMemory (load_cmd_addr, 
                                                                            prefer_file_cache,
                                                                            load_cmd_data_sp->GetBytes(), 
                                                                            load_cmd_data_sp->GetByteSize(),
                                                                            error);
            
            if (load_cmd_bytes_read == kext_summary.header.sizeofcmds)
            {
                // Set the load command data and also set the correct endian
                // swap settings and the correct address size
                load_command_data->SetData(load_cmd_data_sp, 0, kext_summary.header.sizeofcmds);
                load_command_data->SetByteOrder(data.GetByteOrder());
                load_command_data->SetAddressByteSize(data.GetAddressByteSize());
                return true; // We successfully read the mach_header and the load command data
            }

            return false; // We weren't able to read the load command data
        }
    }
    return false; // We failed the read the mach_header
}


//----------------------------------------------------------------------
// Parse the load commands for an image
//----------------------------------------------------------------------
uint32_t
DynamicLoaderMacOSXKernel::ParseLoadCommands (const DataExtractor& data, OSKextLoadedKextSummary& image_info)
{
    uint32_t offset = 0;
    uint32_t cmd_idx;
    Segment segment;
    image_info.Clear (true);

    for (cmd_idx = 0; cmd_idx < image_info.header.ncmds; cmd_idx++)
    {
        // Clear out any load command specific data from image_info since
        // we are about to read it.

        if (data.ValidOffsetForDataOfSize (offset, sizeof(llvm::MachO::load_command)))
        {
            llvm::MachO::load_command load_cmd;
            uint32_t load_cmd_offset = offset;
            load_cmd.cmd = data.GetU32 (&offset);
            load_cmd.cmdsize = data.GetU32 (&offset);
            switch (load_cmd.cmd)
            {
            case llvm::MachO::LoadCommandSegment32:
                {
                    segment.name.SetTrimmedCStringWithLength ((const char *)data.GetData(&offset, 16), 16);
                    // We are putting 4 uint32_t values 4 uint64_t values so
                    // we have to use multiple 32 bit gets below.
                    segment.vmaddr = data.GetU32 (&offset);
                    segment.vmsize = data.GetU32 (&offset);
                    segment.fileoff = data.GetU32 (&offset);
                    segment.filesize = data.GetU32 (&offset);
                    // Extract maxprot, initprot, nsects and flags all at once
                    data.GetU32(&offset, &segment.maxprot, 4);
                    image_info.segments.push_back (segment);
                }
                break;

            case llvm::MachO::LoadCommandSegment64:
                {
                    segment.name.SetTrimmedCStringWithLength ((const char *)data.GetData(&offset, 16), 16);
                    // Extract vmaddr, vmsize, fileoff, and filesize all at once
                    data.GetU64(&offset, &segment.vmaddr, 4);
                    // Extract maxprot, initprot, nsects and flags all at once
                    data.GetU32(&offset, &segment.maxprot, 4);
                    image_info.segments.push_back (segment);
                }
                break;

            case llvm::MachO::LoadCommandUUID:
                image_info.uuid.SetBytes(data.GetData (&offset, 16));
                break;

            default:
                break;
            }
            // Set offset to be the beginning of the next load command.
            offset = load_cmd_offset + load_cmd.cmdsize;
        }
    }
#if 0
    // No slide in the kernel...
    
    // All sections listed in the dyld image info structure will all
    // either be fixed up already, or they will all be off by a single
    // slide amount that is determined by finding the first segment
    // that is at file offset zero which also has bytes (a file size
    // that is greater than zero) in the object file.
    
    // Determine the slide amount (if any)
    const size_t num_sections = image_info.segments.size();
    for (size_t i = 0; i < num_sections; ++i)
    {
        // Iterate through the object file sections to find the
        // first section that starts of file offset zero and that
        // has bytes in the file...
        if (image_info.segments[i].fileoff == 0 && image_info.segments[i].filesize > 0)
        {
            image_info.slide = image_info.address - image_info.segments[i].vmaddr;
            // We have found the slide amount, so we can exit
            // this for loop.
            break;
        }
    }
#endif
    if (image_info.uuid.IsValid())
    {
        bool did_create = false;
        if (FindTargetModule(image_info, true, &did_create))
        {
            if (did_create)
                image_info.module_create_stop_id = m_process->GetStopID();
        }
    }
    return cmd_idx;
}

//----------------------------------------------------------------------
// Dump a Segment to the file handle provided.
//----------------------------------------------------------------------
void
DynamicLoaderMacOSXKernel::Segment::PutToLog (Log *log, addr_t slide) const
{
    if (log)
    {
        if (slide == 0)
            log->Printf ("\t\t%16s [0x%16.16llx - 0x%16.16llx)", 
                         name.AsCString(""), 
                         vmaddr + slide, 
                         vmaddr + slide + vmsize);
        else
            log->Printf ("\t\t%16s [0x%16.16llx - 0x%16.16llx) slide = 0x%llx", 
                         name.AsCString(""), 
                         vmaddr + slide, 
                         vmaddr + slide + vmsize, 
                         slide);
    }
}

const DynamicLoaderMacOSXKernel::Segment *
DynamicLoaderMacOSXKernel::OSKextLoadedKextSummary::FindSegment (const ConstString &name) const
{
    const size_t num_segments = segments.size();
    for (size_t i=0; i<num_segments; ++i)
    {
        if (segments[i].name == name)
            return &segments[i];
    }
    return NULL;
}


//----------------------------------------------------------------------
// Dump an image info structure to the file handle provided.
//----------------------------------------------------------------------
void
DynamicLoaderMacOSXKernel::OSKextLoadedKextSummary::PutToLog (Log *log) const
{
    if (log == NULL)
        return;
    uint8_t *u = (uint8_t *)uuid.GetBytes();

    if (address == LLDB_INVALID_ADDRESS)
    {
        if (u)
        {
            log->Printf("\t                           uuid=%2.2X%2.2X%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X name='%s' (UNLOADED)",
                        u[ 0], u[ 1], u[ 2], u[ 3],
                        u[ 4], u[ 5], u[ 6], u[ 7],
                        u[ 8], u[ 9], u[10], u[11],
                        u[12], u[13], u[14], u[15],
                        name);
        }
        else
            log->Printf("\t                           name='%s' (UNLOADED)", name);
    }
    else
    {
        if (u)
        {
            log->Printf("\taddress=0x%16.16llx uuid=%2.2X%2.2X%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X name='%s'",
                        address,
                        u[ 0], u[ 1], u[ 2], u[ 3],
                        u[ 4], u[ 5], u[ 6], u[ 7],
                        u[ 8], u[ 9], u[10], u[11],
                        u[12], u[13], u[14], u[15],
                        name);
        }
        else
        {
            log->Printf("\taddress=0x%16.16llx path='%s/%s'", address, name);
        }
        for (uint32_t i=0; i<segments.size(); ++i)
            segments[i].PutToLog(log, 0);
    }
}

//----------------------------------------------------------------------
// Dump the _dyld_all_image_infos members and all current image infos
// that we have parsed to the file handle provided.
//----------------------------------------------------------------------
void
DynamicLoaderMacOSXKernel::PutToLog(Log *log) const
{
    if (log == NULL)
        return;

    Mutex::Locker locker(m_mutex);
    log->Printf("gLoadedKextSummaries = 0x%16.16llx { version=%u, entry_size=%u, entry_count=%u, reserved=%u }",
                m_kext_summary_header_addr.GetFileAddress(),
                m_kext_summary_header.version,
                m_kext_summary_header.entry_size,
                m_kext_summary_header.entry_count,
                m_kext_summary_header.reserved);

    size_t i;
    const size_t count = m_kext_summaries.size();
    if (count > 0)
    {
        log->PutCString("Loaded:");
        for (i = 0; i<count; i++)
            m_kext_summaries[i].PutToLog(log);
    }
}

void
DynamicLoaderMacOSXKernel::PrivateInitialize(Process *process)
{
    DEBUG_PRINTF("DynamicLoaderMacOSXKernel::%s() process state = %s\n", __FUNCTION__, StateAsCString(m_process->GetState()));
    Clear(true);
    m_process = process;
    m_process->GetTarget().GetSectionLoadList().Clear();
}

void
DynamicLoaderMacOSXKernel::SetNotificationBreakpointIfNeeded ()
{
    if (m_break_id == LLDB_INVALID_BREAK_ID)
    {
        DEBUG_PRINTF("DynamicLoaderMacOSXKernel::%s() process state = %s\n", __FUNCTION__, StateAsCString(m_process->GetState()));

        
        const bool internal_bp = false;
        const LazyBool skip_prologue = eLazyBoolNo;
        Breakpoint *bp = m_process->GetTarget().CreateBreakpoint (&m_kernel.module_sp->GetFileSpec(),
                                                                  "OSKextLoadedKextSummariesUpdated",
                                                                  eFunctionNameTypeFull,
                                                                  internal_bp,
                                                                  skip_prologue).get();

        bp->SetCallback (DynamicLoaderMacOSXKernel::BreakpointHitCallback, this, true);
        m_break_id = bp->GetID();
    }
}

//----------------------------------------------------------------------
// Member function that gets called when the process state changes.
//----------------------------------------------------------------------
void
DynamicLoaderMacOSXKernel::PrivateProcessStateChanged (Process *process, StateType state)
{
    DEBUG_PRINTF("DynamicLoaderMacOSXKernel::%s(%s)\n", __FUNCTION__, StateAsCString(state));
    switch (state)
    {
    case eStateConnected:
    case eStateAttaching:
    case eStateLaunching:
    case eStateInvalid:
    case eStateUnloaded:
    case eStateExited:
    case eStateDetached:
        Clear(false);
        break;

    case eStateStopped:
        UpdateIfNeeded();
        break;

    case eStateRunning:
    case eStateStepping:
    case eStateCrashed:
    case eStateSuspended:
        break;

    default:
        break;
    }
}

ThreadPlanSP
DynamicLoaderMacOSXKernel::GetStepThroughTrampolinePlan (Thread &thread, bool stop_others)
{
    ThreadPlanSP thread_plan_sp;
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
        log->Printf ("Could not find symbol for step through.");
    return thread_plan_sp;
}

Error
DynamicLoaderMacOSXKernel::CanLoadImage ()
{
    Error error;
    error.SetErrorString("always unsafe to load or unload shared libraries in the darwin kernel");
    return error;
}

void
DynamicLoaderMacOSXKernel::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
DynamicLoaderMacOSXKernel::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
DynamicLoaderMacOSXKernel::GetPluginNameStatic()
{
    return "dynamic-loader.macosx-kernel";
}

const char *
DynamicLoaderMacOSXKernel::GetPluginDescriptionStatic()
{
    return "Dynamic loader plug-in that watches for shared library loads/unloads in the MacOSX kernel.";
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
DynamicLoaderMacOSXKernel::GetPluginName()
{
    return "DynamicLoaderMacOSXKernel";
}

const char *
DynamicLoaderMacOSXKernel::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
DynamicLoaderMacOSXKernel::GetPluginVersion()
{
    return 1;
}

