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

    SystemRuntimeMacOSX (lldb_private::Process *process);

    virtual
    ~SystemRuntimeMacOSX ();

    void
    Clear (bool clear_process);

    void
    DidAttach ();

    void
    DidLaunch();

    void
    ModulesDidLoad (lldb_private::ModuleList &module_list);

    const std::vector<lldb_private::ConstString> &
    GetExtendedBacktraceTypes ();

    lldb::ThreadSP
    GetExtendedBacktraceThread (lldb::ThreadSP thread, lldb_private::ConstString type);

    // REMOVE THE FOLLOWING 4
    bool 
    SetItemEnqueuedBreakpoint ();

    bool
    DidSetItemEnqueuedBreakpoint () const;

    static bool
    ItemEnqueuedCallback (void *baton, lldb_private::StoppointCallbackContext *context, lldb::user_id_t break_id, lldb::user_id_t break_loc_id);

    bool
    ItemEnqueuedBreakpointHit (lldb_private::StoppointCallbackContext *context, lldb::user_id_t break_id, lldb::user_id_t break_loc_id);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

private:
    struct ArchivedBacktrace {
        uint32_t stop_id;
        bool stop_id_is_valid;
        lldb::queue_id_t libdispatch_queue_id;   // LLDB_INVALID_QUEUE_ID if unavailable
        std::vector<lldb::addr_t> pcs;
    };

    SystemRuntimeMacOSX::ArchivedBacktrace
    GetLibdispatchExtendedBacktrace (lldb::ThreadSP thread);

protected:
    lldb::user_id_t m_break_id;
    mutable lldb_private::Mutex m_mutex;

private:

    void
    ParseLdiHeaders ();

    bool
    LdiHeadersInitialized ();

    lldb::addr_t
    GetQueuesHead ();

    lldb::addr_t
    GetItemsHead ();

    lldb::addr_t
    GetThreadCreatorItem (lldb::ThreadSP thread);

    lldb::tid_t
    GetNewThreadUniqueThreadID (lldb::ThreadSP original_thread_sp);

    void
    SetNewThreadThreadName (lldb::ThreadSP original_thread_sp, lldb::ThreadSP new_extended_thread_sp);

    void
    SetNewThreadQueueName (lldb::ThreadSP original_thread_sp, lldb::ThreadSP new_extended_thread_sp);

    void
    SetNewThreadExtendedBacktraceToken (lldb::ThreadSP original_thread_sp, lldb::ThreadSP new_extended_thread_sp);

    void
    SetNewThreadQueueID (lldb::ThreadSP original_thread_sp, lldb::ThreadSP new_extended_thread_sp);

    struct ldi_queue_offsets {
        uint16_t next;
        uint16_t prev;
        uint16_t queue_id;
        uint16_t current_item_ptr;
    };

    struct ldi_item_offsets {
        uint16_t next;
        uint16_t prev;
        uint16_t type;
        uint16_t identifier;
        uint16_t stop_id;
        uint16_t backtrace_length;
        uint16_t backtrace_ptr;
        uint16_t thread_name_ptr;
        uint16_t queue_name_ptr;
        uint16_t unique_thread_id;
        uint16_t pthread_id;
        uint16_t enqueueing_thread_dispatch_queue_t;
        uint16_t enqueueing_thread_dispatch_block_ptr;
        uint16_t queue_id_from_thread_info;
    };

    struct ldi_header {
        uint16_t                    version;
        uint16_t                    ldi_header_size;
        uint16_t                    initialized;        // 0 means uninitialized
        uint16_t                    queue_size;
        uint16_t                    item_size;
        uint64_t                    queues_head_ptr_address;  // Address of queues head structure
        uint64_t                    items_head_ptr_address;   // Address of items_head
        struct ldi_queue_offsets    queue_offsets;
        struct ldi_item_offsets     item_offsets;
    };

    struct ldi_header   m_ldi_header;

    DISALLOW_COPY_AND_ASSIGN (SystemRuntimeMacOSX);
};

#endif  // liblldb_SystemRuntimeMacOSX_h_
