//===-- SystemRuntimeMacOSX.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SystemRuntimeMacOSX_h_
#define liblldb_SystemRuntimeMacOSX_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>
#include <string>

// Other libraries and framework includes
#include "llvm/Support/MachO.h"

#include "lldb/Target/SystemRuntime.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/UUID.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/QueueItem.h"

#include "AppleGetItemInfoHandler.h"
#include "AppleGetQueuesHandler.h"
#include "AppleGetPendingItemsHandler.h"
#include "AppleGetThreadItemInfoHandler.h"

class SystemRuntimeMacOSX : public lldb_private::SystemRuntime
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::SystemRuntime *
    CreateInstance (lldb_private::Process *process);


    //------------------------------------------------------------------
    // instance methods
    //------------------------------------------------------------------

    SystemRuntimeMacOSX (lldb_private::Process *process);

    virtual
    ~SystemRuntimeMacOSX ();

    void
    Clear (bool clear_process);

    void
    Detach ();

    const std::vector<lldb_private::ConstString> &
    GetExtendedBacktraceTypes ();

    lldb::ThreadSP
    GetExtendedBacktraceThread (lldb::ThreadSP thread, lldb_private::ConstString type);

    lldb::ThreadSP
    GetExtendedBacktraceForQueueItem (lldb::QueueItemSP queue_item_sp, lldb_private::ConstString type);

    lldb::ThreadSP
    GetExtendedBacktraceFromItemRef (lldb::addr_t item_ref);

    void
    PopulateQueueList (lldb_private::QueueList &queue_list);

    void
    PopulateQueuesUsingLibBTR (lldb::addr_t queues_buffer, uint64_t queues_buffer_size, uint64_t count, lldb_private::QueueList &queue_list);

    void
    PopulatePendingQueuesUsingLibBTR (lldb::addr_t items_buffer, uint64_t items_buffer_size, uint64_t count, lldb_private::Queue *queue);

    std::string
    GetQueueNameFromThreadQAddress (lldb::addr_t dispatch_qaddr);

    lldb::queue_id_t
    GetQueueIDFromThreadQAddress (lldb::addr_t dispatch_qaddr);

    void
    PopulatePendingItemsForQueue (lldb_private::Queue *queue);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();


protected:
    lldb::user_id_t m_break_id;
    mutable lldb_private::Mutex m_mutex;

private:

    struct libBacktraceRecording_info {
        uint16_t    queue_info_version;
        uint16_t    queue_info_data_offset;
        uint16_t    item_info_version;
        uint16_t    item_info_data_offset;

        libBacktraceRecording_info () :
            queue_info_version(0),
            queue_info_data_offset(0),
            item_info_version(0),
            item_info_data_offset(0) {}
    };


    // A structure which reflects the data recorded in the
    // libBacktraceRecording introspection_dispatch_item_info_s.
    struct ItemInfo {
        lldb::addr_t    item_that_enqueued_this;
        lldb::addr_t    function_or_block;
        uint64_t        enqueuing_thread_id;
        uint64_t        enqueuing_queue_serialnum;
        uint64_t        target_queue_serialnum;
        uint32_t        enqueuing_callstack_frame_count;
        uint32_t        stop_id;
        std::vector<lldb::addr_t>   enqueuing_callstack;
        std::string                 enqueuing_thread_label;
        std::string                 enqueuing_queue_label;
        std::string                 target_queue_label;
    };

    // The offsets of different fields of the dispatch_queue_t structure in
    // a thread/queue process.
    // Based on libdispatch src/queue_private.h, struct dispatch_queue_offsets_s
    // With dqo_version 1-3, the dqo_label field is a per-queue value and cannot be cached.
    // With dqo_version 4 (Mac OS X 10.9 / iOS 7), dqo_label is a constant value that can be cached.
    struct LibdispatchOffsets
    {
        uint16_t dqo_version;
        uint16_t dqo_label;
        uint16_t dqo_label_size;
        uint16_t dqo_flags;
        uint16_t dqo_flags_size;
        uint16_t dqo_serialnum;
        uint16_t dqo_serialnum_size;
        uint16_t dqo_width;
        uint16_t dqo_width_size;
        uint16_t dqo_running;
        uint16_t dqo_running_size;

        LibdispatchOffsets ()
        {
            dqo_version = UINT16_MAX;
            dqo_flags  = UINT16_MAX;
            dqo_serialnum = UINT16_MAX;
            dqo_label = UINT16_MAX;
            dqo_width = UINT16_MAX;
            dqo_running = UINT16_MAX;
        };

        bool
        IsValid ()
        {
            return dqo_version != UINT16_MAX;
        }

        bool
        LabelIsValid ()
        {
            return dqo_label != UINT16_MAX;
        }
    };

    bool
    BacktraceRecordingHeadersInitialized ();

    void
    ReadLibdispatchOffsetsAddress();

    void
    ReadLibdispatchOffsets ();

    std::vector<lldb::addr_t>
    GetPendingItemRefsForQueue (lldb::addr_t queue);

    ItemInfo
    ExtractItemInfoFromBuffer (lldb_private::DataExtractor &extractor);

    lldb_private::AppleGetQueuesHandler m_get_queues_handler;
    lldb_private::AppleGetPendingItemsHandler m_get_pending_items_handler;
    lldb_private::AppleGetItemInfoHandler m_get_item_info_handler;
    lldb_private::AppleGetThreadItemInfoHandler m_get_thread_item_info_handler;

    lldb::addr_t                        m_page_to_free;
    uint64_t                            m_page_to_free_size;
    libBacktraceRecording_info          m_lib_backtrace_recording_info;
    lldb::addr_t                        m_dispatch_queue_offsets_addr;
    struct LibdispatchOffsets           m_libdispatch_offsets;

    DISALLOW_COPY_AND_ASSIGN (SystemRuntimeMacOSX);
};

#endif  // liblldb_SystemRuntimeMacOSX_h_
