//===-- SystemRuntimeMacOSX.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Section.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ClangUtilityFunction.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "Plugins/Process/Utility/HistoryThread.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"


#include "SystemRuntimeMacOSX.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Create an instance of this class. This function is filled into
// the plugin info class that gets handed out by the plugin factory and
// allows the lldb to instantiate an instance of this class.
//----------------------------------------------------------------------
SystemRuntime *
SystemRuntimeMacOSX::CreateInstance (Process* process)
{
    bool create = false;
    if (!create)
    {
        create = true;
        Module* exe_module = process->GetTarget().GetExecutableModulePointer();
        if (exe_module)
        {
            ObjectFile *object_file = exe_module->GetObjectFile();
            if (object_file)
            {
                create = (object_file->GetStrata() == ObjectFile::eStrataUser);
            }
        }
        
        if (create)
        {
            const llvm::Triple &triple_ref = process->GetTarget().GetArchitecture().GetTriple();
            switch (triple_ref.getOS())
            {
                case llvm::Triple::Darwin:
                case llvm::Triple::MacOSX:
                case llvm::Triple::IOS:
                    create = triple_ref.getVendor() == llvm::Triple::Apple;
                    break;
                default:
                    create = false;
                    break;
            }
        }
    }
    
    if (create)
        return new SystemRuntimeMacOSX (process);
    return NULL;
}

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
SystemRuntimeMacOSX::SystemRuntimeMacOSX (Process* process) :
    SystemRuntime(process),
    m_break_id(LLDB_INVALID_BREAK_ID),
    m_mutex(Mutex::eMutexTypeRecursive)
{
    m_ldi_header.initialized = 0;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SystemRuntimeMacOSX::~SystemRuntimeMacOSX()
{
    Clear (true);
}

//----------------------------------------------------------------------
// Clear out the state of this class.
//----------------------------------------------------------------------
void
SystemRuntimeMacOSX::Clear (bool clear_process)
{
    Mutex::Locker locker(m_mutex);

    if (m_process->IsAlive() && LLDB_BREAK_ID_IS_VALID(m_break_id))
        m_process->ClearBreakpointSiteByID(m_break_id);

    if (clear_process)
        m_process = NULL;
    m_break_id = LLDB_INVALID_BREAK_ID;
    m_ldi_header.initialized = 0;
}


void 
SystemRuntimeMacOSX::DidAttach ()
{
}

void 
SystemRuntimeMacOSX::DidLaunch ()
{
}

void
SystemRuntimeMacOSX::ModulesDidLoad (ModuleList &module_list)
{
}

bool
SystemRuntimeMacOSX::LdiHeadersInitialized ()
{
    ParseLdiHeaders();
    return m_ldi_header.initialized;
}

void
SystemRuntimeMacOSX::ParseLdiHeaders ()
{
    if (m_ldi_header.initialized)
        return;
    static ConstString ldi_header_symbol ("ldi_infos");
    SymbolContextList sc_list;
    if (m_process->GetTarget().GetImages().FindSymbolsWithNameAndType (ldi_header_symbol, eSymbolTypeData, sc_list) > 0)
    {
        SymbolContext sc;
        sc_list.GetContextAtIndex (0, sc);
        AddressRange addr_range;
        sc.GetAddressRange (eSymbolContextSymbol, 0, false, addr_range);

        Error error;
        Address ldi_header_addr = addr_range.GetBaseAddress();
        uint8_t version_buf[6];   // version, ldi_header_size, initialized fields
        DataExtractor data (version_buf, sizeof(version_buf), m_process->GetByteOrder(), m_process->GetAddressByteSize());
        const size_t count = sizeof (version_buf);
        const bool prefer_file_cache = false;
        if (m_process->GetTarget().ReadMemory (ldi_header_addr, prefer_file_cache, version_buf, count, error) == sizeof (version_buf))
        {
            int version, initialized, ldi_header_size;
            offset_t offset = 0;
            version = data.GetU16(&offset);
            ldi_header_size = data.GetU16(&offset);
            initialized = data.GetU16(&offset);
            if (initialized)
            {
                DataBufferHeap ldi_header (ldi_header_size, 0);
                DataExtractor ldi_extractor (ldi_header.GetBytes(), ldi_header.GetByteSize(), m_process->GetByteOrder(), m_process->GetAddressByteSize());
                if (m_process->GetTarget().ReadMemory (ldi_header_addr, prefer_file_cache, ldi_header.GetBytes(), ldi_header.GetByteSize(), error) == ldi_header.GetByteSize())
                {
                    offset = 0;
                    m_ldi_header.version = ldi_extractor.GetU16(&offset);
                    m_ldi_header.ldi_header_size = ldi_extractor.GetU16(&offset);
                    m_ldi_header.initialized = ldi_extractor.GetU16(&offset);
                    m_ldi_header.queue_size = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_size = ldi_extractor.GetU16(&offset);

                    // 6 bytes of padding here
                    offset += 6;

                    m_ldi_header.queues_head_ptr_address = ldi_extractor.GetU64(&offset);
                    m_ldi_header.items_head_ptr_address = ldi_extractor.GetU64(&offset);

                    m_ldi_header.queue_offsets.next = ldi_extractor.GetU16(&offset);
                    m_ldi_header.queue_offsets.prev = ldi_extractor.GetU16(&offset);
                    m_ldi_header.queue_offsets.queue_id = ldi_extractor.GetU16(&offset);
                    m_ldi_header.queue_offsets.current_item_ptr = ldi_extractor.GetU16(&offset);

                    m_ldi_header.item_offsets.next = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.prev = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.type = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.identifier = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.stop_id = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.backtrace_length = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.backtrace_ptr = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.thread_name_ptr = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.queue_name_ptr = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.unique_thread_id = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.pthread_id = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.enqueueing_thread_dispatch_queue_t = ldi_extractor.GetU16(&offset);
                    m_ldi_header.item_offsets.enqueueing_thread_dispatch_block_ptr = ldi_extractor.GetU16(&offset);

                    if (ldi_header.GetByteSize () > offset)
                    {
                        m_ldi_header.item_offsets.queue_id_from_thread_info = ldi_extractor.GetU16(&offset);
                    }
                    else
                    {
                        m_ldi_header.item_offsets.queue_id_from_thread_info = 0xffff;
                    }
                }
            }
        }
    }
}

lldb::addr_t
SystemRuntimeMacOSX::GetQueuesHead ()
{
    if (!LdiHeadersInitialized())
        return LLDB_INVALID_ADDRESS;

    Error error;
    addr_t queues_head = m_process->ReadPointerFromMemory (m_ldi_header.queues_head_ptr_address, error);
    if (error.Success() == false || queues_head == LLDB_INVALID_ADDRESS || queues_head == 0)
        return LLDB_INVALID_ADDRESS;

    return queues_head;
}

lldb::addr_t
SystemRuntimeMacOSX::GetItemsHead ()
{
    if (!LdiHeadersInitialized())
        return LLDB_INVALID_ADDRESS;

    Error error;
    addr_t items_head = m_process->ReadPointerFromMemory (m_ldi_header.items_head_ptr_address, error);
    if (error.Success() == false || items_head == LLDB_INVALID_ADDRESS || items_head == 0)
        return LLDB_INVALID_ADDRESS;

    return items_head;
}

addr_t
SystemRuntimeMacOSX::GetThreadCreatorItem (ThreadSP thread_sp)
{
    addr_t enqueued_item_ptr = thread_sp->GetExtendedBacktraceToken();
    if (enqueued_item_ptr == LLDB_INVALID_ADDRESS)
    {
        if (thread_sp->GetQueueID() == LLDB_INVALID_QUEUE_ID || thread_sp->GetQueueID() == 0)
            return LLDB_INVALID_ADDRESS;
    
        Error error;
        uint64_t this_thread_queue_id = thread_sp->GetQueueID();
    
        addr_t queues_head = GetQueuesHead();
        if (queues_head == LLDB_INVALID_ADDRESS)
            return LLDB_INVALID_ADDRESS;
    
        // Step through the queues_head linked list looking for a queue matching this thread, if any
        uint64_t queue_obj_ptr = queues_head;
        enqueued_item_ptr = LLDB_INVALID_ADDRESS;
    
        while (queue_obj_ptr != 0)
        {
            uint64_t queue_id = m_process->ReadUnsignedIntegerFromMemory (queue_obj_ptr + m_ldi_header.queue_offsets.queue_id, 8, LLDB_INVALID_ADDRESS, error);
            if (error.Success() && queue_id != LLDB_INVALID_ADDRESS)
            {
                if (queue_id == this_thread_queue_id)
                {
                    enqueued_item_ptr = m_process->ReadPointerFromMemory (queue_obj_ptr + m_ldi_header.queue_offsets.current_item_ptr, error);
                    break;
                }
            }
            queue_obj_ptr = m_process->ReadPointerFromMemory (queue_obj_ptr + m_ldi_header.queue_offsets.next, error);
            if (error.Success() == false || queue_obj_ptr == LLDB_INVALID_ADDRESS)
            {
                break;
            }
        }
    }
    
    return enqueued_item_ptr;
}

SystemRuntimeMacOSX::ArchivedBacktrace
SystemRuntimeMacOSX::GetLibdispatchExtendedBacktrace (ThreadSP thread_sp)
{
    ArchivedBacktrace bt;
    bt.stop_id = 0;
    bt.stop_id_is_valid = false;
    bt.libdispatch_queue_id = LLDB_INVALID_QUEUE_ID;

    addr_t enqueued_item_ptr = GetThreadCreatorItem (thread_sp);
    
    if (enqueued_item_ptr == LLDB_INVALID_ADDRESS)
        return bt;

    Error error;
    uint32_t ptr_size = m_process->GetTarget().GetArchitecture().GetAddressByteSize();

    uint32_t backtrace_length = m_process->ReadUnsignedIntegerFromMemory (enqueued_item_ptr + m_ldi_header.item_offsets.backtrace_length, 4, 0, error);
    addr_t pc_array_address = m_process->ReadPointerFromMemory (enqueued_item_ptr + m_ldi_header.item_offsets.backtrace_ptr, error);

    if (backtrace_length == 0 || pc_array_address == LLDB_INVALID_ADDRESS)
        return bt;

    for (uint32_t idx = 0; idx < backtrace_length; idx++)
    {
        addr_t pc_val = m_process->ReadPointerFromMemory (pc_array_address + (ptr_size * idx), error);
        if (error.Success() && pc_val != LLDB_INVALID_ADDRESS)
        {
            bt.pcs.push_back (pc_val);
        }
    }

    return bt;
}

const std::vector<ConstString> &
SystemRuntimeMacOSX::GetExtendedBacktraceTypes ()
{
    if (m_types.size () == 0)
    {
        m_types.push_back(ConstString("libdispatch"));
        m_types.push_back(ConstString("pthread"));
    }
    return m_types;
}

void
SystemRuntimeMacOSX::SetNewThreadQueueName (ThreadSP original_thread_sp, ThreadSP new_extended_thread_sp)
{
    addr_t enqueued_item_ptr = GetThreadCreatorItem (original_thread_sp);

    if (enqueued_item_ptr != LLDB_INVALID_ADDRESS)
    {
        Error error;
        addr_t queue_name_ptr = m_process->ReadPointerFromMemory (enqueued_item_ptr + m_ldi_header.item_offsets.queue_name_ptr, error);
        if (queue_name_ptr != LLDB_INVALID_ADDRESS && error.Success())
        {
            char namebuf[512];
            if (m_process->ReadCStringFromMemory (queue_name_ptr, namebuf, sizeof (namebuf), error) > 0 && error.Success())
            {
                new_extended_thread_sp->SetQueueName (namebuf);
            }
        }
    }
}

void
SystemRuntimeMacOSX::SetNewThreadThreadName (ThreadSP original_thread_sp, ThreadSP new_extended_thread_sp)
{
    addr_t enqueued_item_ptr = GetThreadCreatorItem (original_thread_sp);

    if (enqueued_item_ptr != LLDB_INVALID_ADDRESS)
    {
        Error error;
        addr_t thread_name_ptr = m_process->ReadPointerFromMemory (enqueued_item_ptr + m_ldi_header.item_offsets.thread_name_ptr, error);
        if (thread_name_ptr != LLDB_INVALID_ADDRESS && error.Success())
        {
            char namebuf[512];
            if (m_process->ReadCStringFromMemory (thread_name_ptr, namebuf, sizeof (namebuf), error) > 0 && error.Success())
            {
                new_extended_thread_sp->SetName (namebuf);
            }
        }
    }
}


void
SystemRuntimeMacOSX::SetNewThreadExtendedBacktraceToken (ThreadSP original_thread_sp, ThreadSP new_extended_thread_sp)
{
    addr_t enqueued_item_ptr = GetThreadCreatorItem (original_thread_sp);
    if (enqueued_item_ptr != LLDB_INVALID_ADDRESS)
    {
        Error error;
        uint64_t further_extended_backtrace = m_process->ReadPointerFromMemory (enqueued_item_ptr + m_ldi_header.item_offsets.enqueueing_thread_dispatch_block_ptr, error);
        if (error.Success() && further_extended_backtrace != 0 && further_extended_backtrace != LLDB_INVALID_ADDRESS)
        {
            new_extended_thread_sp->SetExtendedBacktraceToken (further_extended_backtrace);
        }
    }
}

void
SystemRuntimeMacOSX::SetNewThreadQueueID (ThreadSP original_thread_sp, ThreadSP new_extended_thread_sp)
{
    queue_id_t queue_id = LLDB_INVALID_QUEUE_ID;
    addr_t enqueued_item_ptr = GetThreadCreatorItem (original_thread_sp);
    if (enqueued_item_ptr != LLDB_INVALID_ADDRESS && m_ldi_header.item_offsets.queue_id_from_thread_info != 0xffff)
    {
        Error error;
        queue_id = m_process->ReadUnsignedIntegerFromMemory (enqueued_item_ptr + m_ldi_header.item_offsets.queue_id_from_thread_info, 8, LLDB_INVALID_QUEUE_ID, error);
        if (!error.Success())
            queue_id = LLDB_INVALID_QUEUE_ID;
    }

    if (queue_id != LLDB_INVALID_QUEUE_ID)
    {
        new_extended_thread_sp->SetQueueID (queue_id);
    }
}


lldb::tid_t
SystemRuntimeMacOSX::GetNewThreadUniqueThreadID (ThreadSP original_thread_sp)
{
    tid_t ret = LLDB_INVALID_THREAD_ID;
    addr_t enqueued_item_ptr = GetThreadCreatorItem (original_thread_sp);
    if (enqueued_item_ptr != LLDB_INVALID_ADDRESS)
    {
        Error error;
        ret = m_process->ReadUnsignedIntegerFromMemory (enqueued_item_ptr + m_ldi_header.item_offsets.unique_thread_id, 8, LLDB_INVALID_THREAD_ID, error);
        if (!error.Success())
            ret = LLDB_INVALID_THREAD_ID;
    }
    return ret;
}

ThreadSP
SystemRuntimeMacOSX::GetExtendedBacktraceThread (ThreadSP original_thread_sp, ConstString type)
{
    ThreadSP new_extended_thread_sp;

    if (type != ConstString("libdispatch"))
        return new_extended_thread_sp;

    ArchivedBacktrace bt = GetLibdispatchExtendedBacktrace (original_thread_sp);

    if (bt.pcs.size() == 0)
        return new_extended_thread_sp;

    tid_t unique_thread_id = GetNewThreadUniqueThreadID (original_thread_sp);

    new_extended_thread_sp.reset (new HistoryThread (*m_process, unique_thread_id, bt.pcs, bt.stop_id, bt.stop_id_is_valid));

    SetNewThreadThreadName (original_thread_sp, new_extended_thread_sp);
    SetNewThreadQueueName (original_thread_sp, new_extended_thread_sp);
    SetNewThreadQueueID (original_thread_sp, new_extended_thread_sp);
    SetNewThreadExtendedBacktraceToken (original_thread_sp, new_extended_thread_sp);
    return new_extended_thread_sp;
}


void
SystemRuntimeMacOSX::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
SystemRuntimeMacOSX::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


lldb_private::ConstString
SystemRuntimeMacOSX::GetPluginNameStatic()
{
    static ConstString g_name("systemruntime-macosx");
    return g_name;
}

const char *
SystemRuntimeMacOSX::GetPluginDescriptionStatic()
{
    return "System runtime plugin for Mac OS X native libraries.";
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
SystemRuntimeMacOSX::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
SystemRuntimeMacOSX::GetPluginVersion()
{
    return 1;
}
