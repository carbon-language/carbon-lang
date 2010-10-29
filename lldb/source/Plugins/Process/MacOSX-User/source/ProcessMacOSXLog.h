//===-- ProcessMacOSXLog.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessMacOSXLog_h_
#define liblldb_ProcessMacOSXLog_h_

// C Includes
// C++ Includes
// Other libraries and framework includes

// Project includes
#include "lldb/Core/Log.h"

#define PD_LOG_VERBOSE                  (1u << 0)
#define PD_LOG_PROCESS                  (1u << 1)
#define PD_LOG_THREAD                   (1u << 2)
#define PD_LOG_EXCEPTIONS               (1u << 3)
#define PD_LOG_MEMORY                   (1u << 4)    // Log memory reads/writes calls
#define PD_LOG_MEMORY_DATA_SHORT        (1u << 5)    // Log short memory reads/writes bytes
#define PD_LOG_MEMORY_DATA_LONG         (1u << 6)    // Log all memory reads/writes bytes
#define PD_LOG_MEMORY_PROTECTIONS       (1u << 7)    // Log memory protection changes
#define PD_LOG_BREAKPOINTS              (1u << 8)
#define PD_LOG_WATCHPOINTS              (1u << 9)
#define PD_LOG_STEP                     (1u << 10)
#define PD_LOG_TASK                     (1u << 11)
#define PD_LOG_ALL                      (UINT32_MAX)
#define PD_LOG_DEFAULT                  (PD_LOG_PROCESS     |\
                                         PD_LOG_TASK        |\
                                         PD_LOG_THREAD      |\
                                         PD_LOG_EXCEPTIONS  |\
                                         PD_LOG_MEMORY      |\
                                         PD_LOG_MEMORY_DATA_SHORT |\
                                         PD_LOG_BREAKPOINTS |\
                                         PD_LOG_WATCHPOINTS |\
                                         PD_LOG_STEP        )

class ProcessMacOSXLog
{
public:
    static lldb_private::Log *
    GetLogIfAllCategoriesSet(uint32_t mask = 0);

    static void
    DisableLog (lldb_private::Args &args, lldb_private::Stream *feedback_strm);

    static void
    DeleteLog ();

    static lldb_private::Log *
    EnableLog (lldb::StreamSP &log_stream_sp, uint32_t log_options, lldb_private::Args &args, lldb_private::Stream *feedback_strm);

    static void
    ListLogCategories (lldb_private::Stream *strm);

    static void
    LogIf (uint32_t mask, const char *format, ...);
};

#endif  // liblldb_ProcessMacOSXLog_h_
