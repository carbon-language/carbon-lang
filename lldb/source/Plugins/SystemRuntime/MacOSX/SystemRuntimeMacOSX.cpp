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
#include "lldb/Target/Queue.h"
#include "lldb/Target/QueueList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Process.h"


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
    m_mutex(Mutex::eMutexTypeRecursive),
    m_get_queues_handler(process),
    m_get_pending_items_handler(process),
    m_get_item_info_handler(process),
    m_get_thread_item_info_handler(process),
    m_page_to_free(LLDB_INVALID_ADDRESS),
    m_page_to_free_size(0),
    m_lib_backtrace_recording_info(),
    m_dispatch_queue_offsets_addr (LLDB_INVALID_ADDRESS),
    m_libdispatch_offsets()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SystemRuntimeMacOSX::~SystemRuntimeMacOSX()
{
    Clear (true);
}

void
SystemRuntimeMacOSX::Detach ()
{
        m_get_queues_handler.Detach();
        m_get_pending_items_handler.Detach();
        m_get_item_info_handler.Detach();
        m_get_thread_item_info_handler.Detach();
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
}


std::string
SystemRuntimeMacOSX::GetQueueNameFromThreadQAddress (addr_t dispatch_qaddr)
{
    std::string dispatch_queue_name;
    if (dispatch_qaddr == LLDB_INVALID_ADDRESS || dispatch_qaddr == 0)
        return "";

    ReadLibdispatchOffsets ();
    if (m_libdispatch_offsets.IsValid ())
    {
        // dispatch_qaddr is from a thread_info(THREAD_IDENTIFIER_INFO) call for a thread -
        // deref it to get the address of the dispatch_queue_t structure for this thread's 
        // queue.
        Error error;
        addr_t dispatch_queue_addr = m_process->ReadPointerFromMemory (dispatch_qaddr, error);
        if (error.Success())
        {
            if (m_libdispatch_offsets.dqo_version >= 4)
            {
                // libdispatch versions 4+, pointer to dispatch name is in the
                // queue structure.
                addr_t pointer_to_label_address = dispatch_queue_addr + m_libdispatch_offsets.dqo_label;
                addr_t label_addr = m_process->ReadPointerFromMemory (pointer_to_label_address, error);
                if (error.Success())
                {
                    m_process->ReadCStringFromMemory (label_addr, dispatch_queue_name, error);
                }
            }
            else
            {
                // libdispatch versions 1-3, dispatch name is a fixed width char array
                // in the queue structure.
                addr_t label_addr = dispatch_queue_addr + m_libdispatch_offsets.dqo_label;
                dispatch_queue_name.resize (m_libdispatch_offsets.dqo_label_size, '\0');
                size_t bytes_read = m_process->ReadMemory (label_addr, &dispatch_queue_name[0], m_libdispatch_offsets.dqo_label_size, error);
                if (bytes_read < m_libdispatch_offsets.dqo_label_size)
                    dispatch_queue_name.erase (bytes_read);
            }
        }
    }
    return dispatch_queue_name;
}

lldb::queue_id_t
SystemRuntimeMacOSX::GetQueueIDFromThreadQAddress (lldb::addr_t dispatch_qaddr)
{
    queue_id_t queue_id = LLDB_INVALID_QUEUE_ID;

    if (dispatch_qaddr == LLDB_INVALID_ADDRESS || dispatch_qaddr == 0)
        return queue_id;

    ReadLibdispatchOffsets ();
    if (m_libdispatch_offsets.IsValid ())
    {
        // dispatch_qaddr is from a thread_info(THREAD_IDENTIFIER_INFO) call for a thread -
        // deref it to get the address of the dispatch_queue_t structure for this thread's 
        // queue.
        Error error;
        uint64_t dispatch_queue_addr = m_process->ReadPointerFromMemory (dispatch_qaddr, error);
        if (error.Success())
        {
            addr_t serialnum_address = dispatch_queue_addr + m_libdispatch_offsets.dqo_serialnum;
            queue_id_t serialnum = m_process->ReadUnsignedIntegerFromMemory (serialnum_address, m_libdispatch_offsets.dqo_serialnum_size, LLDB_INVALID_QUEUE_ID, error);
            if (error.Success())
            {
                queue_id = serialnum;
            }
        }
    }

    return queue_id;
}


void
SystemRuntimeMacOSX::ReadLibdispatchOffsetsAddress ()
{
    if (m_dispatch_queue_offsets_addr != LLDB_INVALID_ADDRESS)
        return;

    static ConstString g_dispatch_queue_offsets_symbol_name ("dispatch_queue_offsets");
    const Symbol *dispatch_queue_offsets_symbol = NULL;

    // libdispatch symbols were in libSystem.B.dylib up through Mac OS X 10.6 ("Snow Leopard")
    ModuleSpec libSystem_module_spec (FileSpec("libSystem.B.dylib", false));
    ModuleSP module_sp(m_process->GetTarget().GetImages().FindFirstModule (libSystem_module_spec));
    if (module_sp)
        dispatch_queue_offsets_symbol = module_sp->FindFirstSymbolWithNameAndType (g_dispatch_queue_offsets_symbol_name, eSymbolTypeData);
    
    // libdispatch symbols are in their own dylib as of Mac OS X 10.7 ("Lion") and later
    if (dispatch_queue_offsets_symbol == NULL)
    {
        ModuleSpec libdispatch_module_spec (FileSpec("libdispatch.dylib", false));
        module_sp = m_process->GetTarget().GetImages().FindFirstModule (libdispatch_module_spec);
        if (module_sp)
            dispatch_queue_offsets_symbol = module_sp->FindFirstSymbolWithNameAndType (g_dispatch_queue_offsets_symbol_name, eSymbolTypeData);
    }
    if (dispatch_queue_offsets_symbol)
        m_dispatch_queue_offsets_addr = dispatch_queue_offsets_symbol->GetAddress().GetLoadAddress(&m_process->GetTarget());
}

void
SystemRuntimeMacOSX::ReadLibdispatchOffsets ()
{
    if (m_libdispatch_offsets.IsValid())
        return;

    ReadLibdispatchOffsetsAddress ();

    uint8_t memory_buffer[sizeof (struct LibdispatchOffsets)];
    DataExtractor data (memory_buffer, 
                        sizeof(memory_buffer), 
                        m_process->GetByteOrder(), 
                        m_process->GetAddressByteSize());

    Error error;
    if (m_process->ReadMemory (m_dispatch_queue_offsets_addr, memory_buffer, sizeof(memory_buffer), error) == sizeof(memory_buffer))
    {
        lldb::offset_t data_offset = 0;

        // The struct LibdispatchOffsets is a series of uint16_t's - extract them all
        // in one big go.
        data.GetU16 (&data_offset, &m_libdispatch_offsets.dqo_version, sizeof (struct LibdispatchOffsets) / sizeof (uint16_t));
    }
}


ThreadSP
SystemRuntimeMacOSX::GetExtendedBacktraceThread (ThreadSP real_thread, ConstString type)
{
    ThreadSP originating_thread_sp;
    if (BacktraceRecordingHeadersInitialized() && type == ConstString ("libdispatch"))
    {
        Error error;

        // real_thread is either an actual, live thread (in which case we need to call into
        // libBacktraceRecording to find its originator) or it is an extended backtrace itself,
        // in which case we get the token from it and call into libBacktraceRecording to find
        // the originator of that token.

        if (real_thread->GetExtendedBacktraceToken() != LLDB_INVALID_ADDRESS)
        {
            originating_thread_sp = GetExtendedBacktraceFromItemRef (real_thread->GetExtendedBacktraceToken());
        }
        else
        {
            ThreadSP cur_thread_sp (m_process->GetThreadList().GetSelectedThread());
            AppleGetThreadItemInfoHandler::GetThreadItemInfoReturnInfo ret = m_get_thread_item_info_handler.GetThreadItemInfo (*cur_thread_sp.get(), real_thread->GetID(), m_page_to_free, m_page_to_free_size, error);
            m_page_to_free = LLDB_INVALID_ADDRESS;
            m_page_to_free_size = 0;
            if (ret.item_buffer_ptr != 0 &&  ret.item_buffer_ptr != LLDB_INVALID_ADDRESS && ret.item_buffer_size > 0)
            {
                DataBufferHeap data (ret.item_buffer_size, 0);
                if (m_process->ReadMemory (ret.item_buffer_ptr, data.GetBytes(), ret.item_buffer_size, error) && error.Success())
                {
                    DataExtractor extractor (data.GetBytes(), data.GetByteSize(), m_process->GetByteOrder(), m_process->GetAddressByteSize());
                    ItemInfo item = ExtractItemInfoFromBuffer (extractor);
                    bool stop_id_is_valid = true;
                    if (item.stop_id == 0)
                        stop_id_is_valid = false;
                    originating_thread_sp.reset (new HistoryThread (*m_process,
                                                                    item.enqueuing_thread_id,
                                                                    item.enqueuing_callstack,
                                                                    item.stop_id,
                                                                    stop_id_is_valid));
                    originating_thread_sp->SetExtendedBacktraceToken (item.item_that_enqueued_this);
                    originating_thread_sp->SetQueueName (item.enqueuing_queue_label.c_str());
                    originating_thread_sp->SetQueueID (item.enqueuing_queue_serialnum);
//                    originating_thread_sp->SetThreadName (item.enqueuing_thread_label.c_str());
                }
                m_page_to_free = ret.item_buffer_ptr;
                m_page_to_free_size = ret.item_buffer_size;
            }
        }
    }
    return originating_thread_sp;
}

ThreadSP
SystemRuntimeMacOSX::GetExtendedBacktraceFromItemRef (lldb::addr_t item_ref)
{
    ThreadSP return_thread_sp;

    AppleGetItemInfoHandler::GetItemInfoReturnInfo ret;
    ThreadSP cur_thread_sp (m_process->GetThreadList().GetSelectedThread());
    Error error;
    ret = m_get_item_info_handler.GetItemInfo (*cur_thread_sp.get(), item_ref, m_page_to_free, m_page_to_free_size, error);
    m_page_to_free = LLDB_INVALID_ADDRESS;
    m_page_to_free_size = 0;
    if (ret.item_buffer_ptr != 0 &&  ret.item_buffer_ptr != LLDB_INVALID_ADDRESS && ret.item_buffer_size > 0)
    {
        DataBufferHeap data (ret.item_buffer_size, 0);
        if (m_process->ReadMemory (ret.item_buffer_ptr, data.GetBytes(), ret.item_buffer_size, error) && error.Success())
        {
            DataExtractor extractor (data.GetBytes(), data.GetByteSize(), m_process->GetByteOrder(), m_process->GetAddressByteSize());
            ItemInfo item = ExtractItemInfoFromBuffer (extractor);
            bool stop_id_is_valid = true;
            if (item.stop_id == 0)
                stop_id_is_valid = false;
            return_thread_sp.reset (new HistoryThread (*m_process,
                                                            item.enqueuing_thread_id,
                                                            item.enqueuing_callstack,
                                                            item.stop_id,
                                                            stop_id_is_valid));
            return_thread_sp->SetExtendedBacktraceToken (item.item_that_enqueued_this);
            return_thread_sp->SetQueueName (item.enqueuing_queue_label.c_str());
            return_thread_sp->SetQueueID (item.enqueuing_queue_serialnum);
//            return_thread_sp->SetThreadName (item.enqueuing_thread_label.c_str());

            m_page_to_free = ret.item_buffer_ptr;
            m_page_to_free_size = ret.item_buffer_size;
        }
    }
    return return_thread_sp;
}

ThreadSP
SystemRuntimeMacOSX::GetExtendedBacktraceForQueueItem (QueueItemSP queue_item_sp, ConstString type)
{
    ThreadSP extended_thread_sp;
    if (type != ConstString("libdispatch"))
        return extended_thread_sp;

    bool stop_id_is_valid = true;
    if (queue_item_sp->GetStopID() == 0)
        stop_id_is_valid = false;

    extended_thread_sp.reset (new HistoryThread (*m_process, 
                                                 queue_item_sp->GetEnqueueingThreadID(),
                                                 queue_item_sp->GetEnqueueingBacktrace(),
                                                 queue_item_sp->GetStopID(),
                                                 stop_id_is_valid));
    extended_thread_sp->SetExtendedBacktraceToken (queue_item_sp->GetItemThatEnqueuedThis());
    extended_thread_sp->SetQueueName (queue_item_sp->GetQueueLabel().c_str());
    extended_thread_sp->SetQueueID (queue_item_sp->GetEnqueueingQueueID());
//    extended_thread_sp->SetThreadName (queue_item_sp->GetThreadLabel().c_str());

    return extended_thread_sp;
}

/* Returns true if we were able to get the version / offset information
 * out of libBacktraceRecording.  false means we were unable to retrieve
 * this; the queue_info_version field will be 0.
 */

bool
SystemRuntimeMacOSX::BacktraceRecordingHeadersInitialized ()
{
    if (m_lib_backtrace_recording_info.queue_info_version != 0)
        return true;

    addr_t queue_info_version_address = LLDB_INVALID_ADDRESS;
    addr_t queue_info_data_offset_address = LLDB_INVALID_ADDRESS;
    addr_t item_info_version_address = LLDB_INVALID_ADDRESS;
    addr_t item_info_data_offset_address = LLDB_INVALID_ADDRESS;
    Target &target = m_process->GetTarget();


    static ConstString introspection_dispatch_queue_info_version ("__introspection_dispatch_queue_info_version");
    SymbolContextList sc_list;
    if (m_process->GetTarget().GetImages().FindSymbolsWithNameAndType (introspection_dispatch_queue_info_version, eSymbolTypeData, sc_list) > 0)
    {
        SymbolContext sc;
        sc_list.GetContextAtIndex (0, sc);
        AddressRange addr_range;
        sc.GetAddressRange (eSymbolContextSymbol, 0, false, addr_range);
        queue_info_version_address = addr_range.GetBaseAddress().GetLoadAddress(&target);
    }
    sc_list.Clear();

    static ConstString introspection_dispatch_queue_info_data_offset ("__introspection_dispatch_queue_info_data_offset");
    if (m_process->GetTarget().GetImages().FindSymbolsWithNameAndType (introspection_dispatch_queue_info_data_offset, eSymbolTypeData, sc_list) > 0)
    {
        SymbolContext sc;
        sc_list.GetContextAtIndex (0, sc);
        AddressRange addr_range;
        sc.GetAddressRange (eSymbolContextSymbol, 0, false, addr_range);
        queue_info_data_offset_address = addr_range.GetBaseAddress().GetLoadAddress(&target);
    }
    sc_list.Clear();

    static ConstString introspection_dispatch_item_info_version ("__introspection_dispatch_item_info_version");
    if (m_process->GetTarget().GetImages().FindSymbolsWithNameAndType (introspection_dispatch_item_info_version, eSymbolTypeData, sc_list) > 0)
    {
        SymbolContext sc;
        sc_list.GetContextAtIndex (0, sc);
        AddressRange addr_range;
        sc.GetAddressRange (eSymbolContextSymbol, 0, false, addr_range);
        item_info_version_address = addr_range.GetBaseAddress().GetLoadAddress(&target);
    }
    sc_list.Clear();

    static ConstString introspection_dispatch_item_info_data_offset ("__introspection_dispatch_item_info_data_offset");
    if (m_process->GetTarget().GetImages().FindSymbolsWithNameAndType (introspection_dispatch_item_info_data_offset, eSymbolTypeData, sc_list) > 0)
    {
        SymbolContext sc;
        sc_list.GetContextAtIndex (0, sc);
        AddressRange addr_range;
        sc.GetAddressRange (eSymbolContextSymbol, 0, false, addr_range);
        item_info_data_offset_address = addr_range.GetBaseAddress().GetLoadAddress(&target);
    }

    if (queue_info_version_address != LLDB_INVALID_ADDRESS
        && queue_info_data_offset_address != LLDB_INVALID_ADDRESS
        && item_info_version_address != LLDB_INVALID_ADDRESS
        && item_info_data_offset_address != LLDB_INVALID_ADDRESS)
    {
        Error error;
        m_lib_backtrace_recording_info.queue_info_version = m_process->ReadUnsignedIntegerFromMemory (queue_info_version_address, 2, 0, error);
        if (error.Success())
        {
            m_lib_backtrace_recording_info.queue_info_data_offset = m_process->ReadUnsignedIntegerFromMemory (queue_info_data_offset_address, 2, 0, error);
            if (error.Success())
            {
                m_lib_backtrace_recording_info.item_info_version = m_process->ReadUnsignedIntegerFromMemory (item_info_version_address, 2, 0, error);
                if (error.Success())
                {
                    m_lib_backtrace_recording_info.item_info_data_offset = m_process->ReadUnsignedIntegerFromMemory (item_info_data_offset_address, 2, 0, error);
                    if (!error.Success())
                    {
                        m_lib_backtrace_recording_info.queue_info_version = 0;
                    }
                }
                else
                {
                    m_lib_backtrace_recording_info.queue_info_version = 0;
                }
            }
            else
            {
                m_lib_backtrace_recording_info.queue_info_version = 0;
            }
        }
    }

    return m_lib_backtrace_recording_info.queue_info_version != 0;
}

const std::vector<ConstString> &
SystemRuntimeMacOSX::GetExtendedBacktraceTypes ()
{
    if (m_types.size () == 0)
    {
        m_types.push_back(ConstString("libdispatch"));
        // We could have pthread as another type in the future if we have a way of
        // gathering that information & it's useful to distinguish between them.
    }
    return m_types;
}

void
SystemRuntimeMacOSX::PopulateQueueList (lldb_private::QueueList &queue_list)
{
    if (!BacktraceRecordingHeadersInitialized())
    {
        // We don't have libBacktraceRecording -- build the list of queues by looking at
        // all extant threads, and the queues that they currently belong to.

        for (ThreadSP thread_sp : m_process->Threads())
        {
            if (thread_sp->GetQueueID() != LLDB_INVALID_QUEUE_ID)
            {
                if (queue_list.FindQueueByID (thread_sp->GetQueueID()).get() == NULL)
                {
                    QueueSP queue_sp (new Queue(m_process->shared_from_this(), thread_sp->GetQueueID(), thread_sp->GetQueueName()));
                    queue_list.AddQueue (queue_sp);
                }
            }
        }
    }
    else
    {
        AppleGetQueuesHandler::GetQueuesReturnInfo queue_info_pointer;
        ThreadSP cur_thread_sp (m_process->GetThreadList().GetSelectedThread());
        if (cur_thread_sp)
        { 
            Error error;
            queue_info_pointer = m_get_queues_handler.GetCurrentQueues (*cur_thread_sp.get(), m_page_to_free, m_page_to_free_size, error);
            m_page_to_free = LLDB_INVALID_ADDRESS;
            m_page_to_free_size = 0;
            if (error.Success())
            {

                if (queue_info_pointer.count > 0 
                    && queue_info_pointer.queues_buffer_size > 0
                    && queue_info_pointer.queues_buffer_ptr != 0 
                    && queue_info_pointer.queues_buffer_ptr != LLDB_INVALID_ADDRESS)
                {
                    PopulateQueuesUsingLibBTR (queue_info_pointer.queues_buffer_ptr, queue_info_pointer.queues_buffer_size, queue_info_pointer.count, queue_list);
                }
            }
        }
    }
}

void
SystemRuntimeMacOSX::PopulatePendingItemsForQueue (Queue *queue)
{
    if (BacktraceRecordingHeadersInitialized())
    {
        std::vector<addr_t> pending_item_refs = GetPendingItemRefsForQueue (queue->GetLibdispatchQueueAddress());
        for (addr_t pending_item : pending_item_refs)
        {
            AppleGetItemInfoHandler::GetItemInfoReturnInfo ret;
            ThreadSP cur_thread_sp (m_process->GetThreadList().GetSelectedThread());
            Error error;
            ret = m_get_item_info_handler.GetItemInfo (*cur_thread_sp.get(), pending_item, m_page_to_free, m_page_to_free_size, error);
            m_page_to_free = LLDB_INVALID_ADDRESS;
            m_page_to_free_size = 0;
            if (ret.item_buffer_ptr != 0 &&  ret.item_buffer_ptr != LLDB_INVALID_ADDRESS && ret.item_buffer_size > 0)
            {
                DataBufferHeap data (ret.item_buffer_size, 0);
                if (m_process->ReadMemory (ret.item_buffer_ptr, data.GetBytes(), ret.item_buffer_size, error) && error.Success())
                {
                    DataExtractor extractor (data.GetBytes(), data.GetByteSize(), m_process->GetByteOrder(), m_process->GetAddressByteSize());
                    ItemInfo item = ExtractItemInfoFromBuffer (extractor);
                    QueueItemSP queue_item_sp (new QueueItem (queue->shared_from_this()));
                    queue_item_sp->SetItemThatEnqueuedThis (item.item_that_enqueued_this);

                    Address addr;
                    if (!m_process->GetTarget().ResolveLoadAddress (item.function_or_block, addr, item.stop_id))
                    {
                        m_process->GetTarget().ResolveLoadAddress (item.function_or_block, addr);
                    }
                    queue_item_sp->SetAddress (addr);
                    queue_item_sp->SetEnqueueingThreadID (item.enqueuing_thread_id);
                    queue_item_sp->SetTargetQueueID (item.enqueuing_thread_id);
                    queue_item_sp->SetStopID (item.stop_id);
                    queue_item_sp->SetEnqueueingBacktrace (item.enqueuing_callstack);
                    queue_item_sp->SetThreadLabel (item.enqueuing_thread_label);
                    queue_item_sp->SetQueueLabel (item.enqueuing_queue_label);
                    queue_item_sp->SetTargetQueueLabel (item.target_queue_label);

                    queue->PushPendingQueueItem (queue_item_sp);
                }
            }
        }
    }
}

// Returns an array of introspection_dispatch_item_info_ref's for the pending items on
// a queue.  The information about each of these pending items then needs to be fetched
// individually by passing the ref to libBacktraceRecording.

std::vector<lldb::addr_t>
SystemRuntimeMacOSX::GetPendingItemRefsForQueue (lldb::addr_t queue)
{
    std::vector<addr_t> pending_item_refs;
    AppleGetPendingItemsHandler::GetPendingItemsReturnInfo pending_items_pointer;
    ThreadSP cur_thread_sp (m_process->GetThreadList().GetSelectedThread());
    if (cur_thread_sp)
    { 
        Error error;
        pending_items_pointer = m_get_pending_items_handler.GetPendingItems (*cur_thread_sp.get(), queue, m_page_to_free, m_page_to_free_size, error);
        m_page_to_free = LLDB_INVALID_ADDRESS;
        m_page_to_free_size = 0;
        if (error.Success())
        {
            if (pending_items_pointer.count > 0
                && pending_items_pointer.items_buffer_size > 0
                && pending_items_pointer.items_buffer_ptr != 0
                && pending_items_pointer.items_buffer_ptr != LLDB_INVALID_ADDRESS)
            {
                DataBufferHeap data (pending_items_pointer.items_buffer_size, 0);
                if (m_process->ReadMemory (pending_items_pointer.items_buffer_ptr, data.GetBytes(), pending_items_pointer.items_buffer_size, error))
                {
                    offset_t offset = 0;
                    DataExtractor extractor (data.GetBytes(), data.GetByteSize(), m_process->GetByteOrder(), m_process->GetAddressByteSize());
                    int i = 0;
                    while (offset < pending_items_pointer.items_buffer_size && i < pending_items_pointer.count)
                    {
                        pending_item_refs.push_back (extractor.GetPointer (&offset));
                        i++;
                    }
                }
                m_page_to_free = pending_items_pointer.items_buffer_ptr;
                m_page_to_free_size = pending_items_pointer.items_buffer_size;
            }
        }
    }
    return pending_item_refs;
}


void
SystemRuntimeMacOSX::PopulateQueuesUsingLibBTR (lldb::addr_t queues_buffer, uint64_t queues_buffer_size, 
                                                uint64_t count, lldb_private::QueueList &queue_list)
{
    Error error;
    DataBufferHeap data (queues_buffer_size, 0);
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SYSTEM_RUNTIME));
    if (m_process->ReadMemory (queues_buffer, data.GetBytes(), queues_buffer_size, error) == queues_buffer_size && error.Success())
    {
        // We've read the information out of inferior memory; free it on the next call we make
        m_page_to_free = queues_buffer;
        m_page_to_free_size = queues_buffer_size;

        DataExtractor extractor (data.GetBytes(), data.GetByteSize(), m_process->GetByteOrder(), m_process->GetAddressByteSize());
        offset_t offset = 0;
        uint64_t queues_read = 0;

        // The information about the queues is stored in this format (v1):
        // typedef struct introspection_dispatch_queue_info_s {
        //     uint32_t offset_to_next;
        //     dispatch_queue_t queue;
        //     uint64_t serialnum;     // queue's serialnum in the process, as provided by libdispatch
        //     uint32_t running_work_items_count;
        //     uint32_t pending_work_items_count;
        // 
        //     char data[];     // Starting here, we have variable-length data:
        //     // char queue_label[];
        // } introspection_dispatch_queue_info_s;

        while (queues_read < count && offset < queues_buffer_size)
        {
            offset_t    start_of_this_item = offset;

            uint32_t    offset_to_next = extractor.GetU32 (&offset);

            offset += 4; // Skip over the 4 bytes of reserved space
            addr_t      queue = extractor.GetPointer (&offset);
            uint64_t    serialnum = extractor.GetU64 (&offset);
            uint32_t    running_work_items_count = extractor.GetU32 (&offset);
            uint32_t    pending_work_items_count = extractor.GetU32 (&offset);

            // Read the first field of the variable length data
            offset = start_of_this_item + m_lib_backtrace_recording_info.queue_info_data_offset;
            const char *queue_label = extractor.GetCStr (&offset);
            if (queue_label == NULL)
                queue_label = "";

            offset_t    start_of_next_item = start_of_this_item + offset_to_next;
            offset = start_of_next_item;

            if (log)
                log->Printf ("SystemRuntimeMacOSX::PopulateQueuesUsingLibBTR added queue with dispatch_queue_t 0x%" PRIx64 ", serial number 0x%" PRIx64 ", running items %d, pending items %d, name '%s'", queue, serialnum, running_work_items_count, pending_work_items_count, queue_label);

            QueueSP queue_sp (new Queue (m_process->shared_from_this(), serialnum, queue_label));
            queue_sp->SetNumRunningWorkItems (running_work_items_count);
            queue_sp->SetNumPendingWorkItems (pending_work_items_count);
            queue_sp->SetLibdispatchQueueAddress (queue);
            queue_list.AddQueue (queue_sp);
            queues_read++;
        }
    }
}

SystemRuntimeMacOSX::ItemInfo
SystemRuntimeMacOSX::ExtractItemInfoFromBuffer (lldb_private::DataExtractor &extractor)
{
    ItemInfo item;

    offset_t offset = 0;
    
    item.item_that_enqueued_this = extractor.GetPointer (&offset);
    item.function_or_block = extractor.GetPointer (&offset);
    item.enqueuing_thread_id = extractor.GetU64 (&offset);
    item.enqueuing_queue_serialnum = extractor.GetU64 (&offset);
    item.target_queue_serialnum = extractor.GetU64 (&offset);
    item.enqueuing_callstack_frame_count = extractor.GetU32 (&offset);
    item.stop_id = extractor.GetU32 (&offset);

    offset = m_lib_backtrace_recording_info.item_info_data_offset;

    for (uint32_t i = 0; i < item.enqueuing_callstack_frame_count; i++)
    {
        item.enqueuing_callstack.push_back (extractor.GetPointer (&offset));
    }
    item.enqueuing_thread_label = extractor.GetCStr (&offset);
    item.enqueuing_queue_label = extractor.GetCStr (&offset);
    item.target_queue_label = extractor.GetCStr (&offset);

    return item;
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
