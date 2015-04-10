//===-- ProcessWindowsLog.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessWindowsLog_h_
#define liblldb_ProcessWindowsLog_h_

#include "lldb/Core/Log.h"

#define WINDOWS_LOG_VERBOSE                  (1u << 0)
#define WINDOWS_LOG_PROCESS                  (1u << 1)
#define WINDOWS_LOG_THREAD                   (1u << 2)
#define WINDOWS_LOG_MEMORY                   (1u << 3)    // Log memory reads/writes calls
#define WINDOWS_LOG_MEMORY_DATA_SHORT        (1u << 4)    // Log short memory reads/writes bytes
#define WINDOWS_LOG_MEMORY_DATA_LONG         (1u << 5)    // Log all memory reads/writes bytes
#define WINDOWS_LOG_BREAKPOINTS              (1u << 6)
#define WINDOWS_LOG_STEP                     (1u << 7)
#define WINDOWS_LOG_ASYNC                    (1u << 8)
#define WINDOWS_LOG_REGISTERS                (1u << 9)
#define WINDOWS_LOG_ALL                      (UINT32_MAX)
#define WINDOWS_LOG_DEFAULT                  WINDOWS_LOG_ASYNC

// The size which determines "short memory reads/writes".
#define WINDOWS_LOG_MEMORY_SHORT_BYTES       (4 * sizeof(ptrdiff_t))

class ProcessWindowsLog
{
    static int m_nestinglevel;
    static const char *m_pluginname;

public:
    // ---------------------------------------------------------------------
    // Public Static Methods
    // ---------------------------------------------------------------------
    static void
    Initialize();

    static void
    RegisterPluginName(const char *pluginName)
    {
        m_pluginname = pluginName;
    }

    static void
    RegisterPluginName(lldb_private::ConstString pluginName)
    {
        m_pluginname = pluginName.GetCString();
    }

    static lldb_private::Log *
    GetLogIfAllCategoriesSet(uint32_t mask = 0);

    static void
    DisableLog(const char **args, lldb_private::Stream *feedback_strm);

    static lldb_private::Log *
    EnableLog(lldb::StreamSP &log_stream_sp, uint32_t log_options,
               const char **args, lldb_private::Stream *feedback_strm);

    static void
    ListLogCategories(lldb_private::Stream *strm);

    static void
    LogIf(uint32_t mask, const char *format, ...);

};

#endif  // liblldb_ProcessWindowsLog_h_
