//===-- MachException.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef liblldb_MachException_h_
#define liblldb_MachException_h_

#include <mach/mach.h>
#include <vector>
#include "lldb/lldb-private.h"
#include "lldb/Target/Thread.h"
// TODO: Get the config script to run to this plug-in
//#include "PDConfig.h"
#define HAVE_64_BIT_MACH_EXCEPTIONS // REMOVE THIS WHEN PDConfig.h is included above
#ifdef HAVE_64_BIT_MACH_EXCEPTIONS

#define MACH_EXCEPTION_DATA_FMT_DEC "%lld"
#define MACH_EXCEPTION_DATA_FMT_HEX "0x%16.16llx"
#define MACH_EXCEPTION_DATA_FMT_MINHEX "0x%llx"

#else

#define MACH_EXCEPTION_DATA_FMT_DEC "%d"
#define MACH_EXCEPTION_DATA_FMT_HEX "0x%8.8x"
#define MACH_EXCEPTION_DATA_FMT_MINHEX "0x%x"

#endif

class MachProcess;

typedef union MachMessageTag
{
    mach_msg_header_t hdr;
    char data[1024];
} MachMessage;


class MachException
{
public:

    struct PortInfo
    {
        exception_mask_t        masks[EXC_TYPES_COUNT];
        mach_port_t             ports[EXC_TYPES_COUNT];
        exception_behavior_t    behaviors[EXC_TYPES_COUNT];
        thread_state_flavor_t   flavors[EXC_TYPES_COUNT];
        mach_msg_type_number_t  count;

        PortInfo();
        kern_return_t   Save(task_t task);
        kern_return_t   Restore(task_t task);
    };

    struct Data
    {
        task_t task_port;
        lldb::tid_t thread_port;
        exception_type_t exc_type;
        std::vector<lldb::addr_t> exc_data;
        Data() :
            task_port(TASK_NULL),
            thread_port(THREAD_NULL),
            exc_type(0),
            exc_data()
            {
            }

        void Clear()
        {
            task_port = TASK_NULL;
            thread_port = THREAD_NULL;
            exc_type = 0;
            exc_data.clear();
        }
        bool IsValid() const
        {
            return  task_port != TASK_NULL &&
                    thread_port != THREAD_NULL &&
                    exc_type != 0;
        }
        // Return the SoftSignal for this MachException data, or zero if there is none
        int SoftSignal() const
        {
            if (exc_type == EXC_SOFTWARE && exc_data.size() == 2 && exc_data[0] == EXC_SOFT_SIGNAL)
                return exc_data[1];
            return LLDB_INVALID_SIGNAL_NUMBER;
        }
        bool IsBreakpoint() const
        {
            return (exc_type == EXC_BREAKPOINT) || ((exc_type == EXC_SOFTWARE) && exc_data[0] == 1);
        }
        void PutToLog(lldb_private::Log *log) const;
        void DumpStopReason() const;
        lldb::StopInfoSP GetStopInfo (lldb_private::Thread &thread) const;
    };

    struct Message
    {
        MachMessage exc_msg;
        MachMessage reply_msg;
        Data state;

        Message() :
            exc_msg(),
            reply_msg(),
            state()
        {
            memset(&exc_msg,   0, sizeof(exc_msg));
            memset(&reply_msg, 0, sizeof(reply_msg));
        }
        bool CatchExceptionRaise();
        void PutToLog(lldb_private::Log *log) const;
        kern_return_t Reply (task_t task, pid_t pid, int signal);
        kern_return_t Receive( mach_port_t receive_port,
                               mach_msg_option_t options,
                               mach_msg_timeout_t timeout,
                               mach_port_t notify_port = MACH_PORT_NULL);

        typedef std::vector<Message>        collection;
        typedef collection::iterator        iterator;
        typedef collection::const_iterator  const_iterator;
    };

    enum
    {
        e_actionForward,    // Forward signal to inferior process
        e_actionStop        // Stop when this signal is received
    };
    struct Action
    {
        task_t task_port;            // Set to TASK_NULL for any TASK
        lldb::tid_t thread_port;        // Set to THREAD_NULL for any thread
        exception_type_t exc_mask;    // Mach exception mask to watch for
        std::vector<mach_exception_data_type_t> exc_data_mask;    // Mask to apply to exception data, or empty to ignore exc_data value for exception
        std::vector<mach_exception_data_type_t> exc_data_value;    // Value to compare to exception data after masking, or empty to ignore exc_data value for exception
        uint8_t flags;                // Action flags describing what to do with the exception
    };
    static const char *Name(exception_type_t exc_type);
};

#endif
