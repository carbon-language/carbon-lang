//===-- ProcessGDBRemoteLog.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProcessGDBRemoteLog.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Core/StreamFile.h"

#include "ProcessGDBRemote.h"

using namespace lldb;
using namespace lldb_private;


static Log* g_log = NULL; // Leak for now as auto_ptr was being cleaned up
                          // by global constructors before other threads
                          // were done with it.
Log *
ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (uint32_t mask)
{
    Log *log = g_log;
    if (log && mask)
    {
        uint32_t log_mask = log->GetMask().Get();
        if ((log_mask & mask) != mask)
            return NULL;
    }
    return log;
}

void
ProcessGDBRemoteLog::DisableLog ()
{
    if (g_log)
    {
        delete g_log;
        g_log = NULL;
    }
}

Log *
ProcessGDBRemoteLog::EnableLog (StreamSP &log_stream_sp, uint32_t log_options, Args &args, Stream *feedback_strm)
{
    DisableLog ();
    g_log = new Log (log_stream_sp);
    if (g_log)
    {
        uint32_t flag_bits = 0;
        bool got_unknown_category = false;
        const size_t argc = args.GetArgumentCount();
        for (size_t i=0; i<argc; ++i)
        {
            const char *arg = args.GetArgumentAtIndex(i);

            if      (::strcasecmp (arg, "all")        == 0   ) flag_bits |= GDBR_LOG_ALL;
            else if (::strcasecmp (arg, "async")      == 0   ) flag_bits |= GDBR_LOG_ASYNC;
            else if (::strcasestr (arg, "break")      == arg ) flag_bits |= GDBR_LOG_BREAKPOINTS;
            else if (::strcasestr (arg, "comm")       == arg ) flag_bits |= GDBR_LOG_COMM;
            else if (::strcasecmp (arg, "default")    == 0   ) flag_bits |= GDBR_LOG_DEFAULT;
            else if (::strcasecmp (arg, "packets")    == 0   ) flag_bits |= GDBR_LOG_PACKETS;
            else if (::strcasecmp (arg, "memory")     == 0   ) flag_bits |= GDBR_LOG_MEMORY;
            else if (::strcasecmp (arg, "data-short") == 0   ) flag_bits |= GDBR_LOG_MEMORY_DATA_SHORT;
            else if (::strcasecmp (arg, "data-long")  == 0   ) flag_bits |= GDBR_LOG_MEMORY_DATA_LONG;
            else if (::strcasecmp (arg, "process")    == 0   ) flag_bits |= GDBR_LOG_PROCESS;
            else if (::strcasecmp (arg, "step")       == 0   ) flag_bits |= GDBR_LOG_STEP;
            else if (::strcasecmp (arg, "thread")     == 0   ) flag_bits |= GDBR_LOG_THREAD;
            else if (::strcasecmp (arg, "verbose")    == 0   ) flag_bits |= GDBR_LOG_VERBOSE;
            else if (::strcasestr (arg, "watch")      == arg ) flag_bits |= GDBR_LOG_WATCHPOINTS;
            else
            {
                feedback_strm->Printf("error: unrecognized log category '%s'\n", arg);
                if (got_unknown_category == false)
                {
                    got_unknown_category = true;
                    ListLogCategories (feedback_strm);
                }
            }
        }
        if (flag_bits == 0)
            flag_bits = GDBR_LOG_DEFAULT;
        g_log->GetMask().Reset(flag_bits);
        g_log->GetOptions().Reset(log_options);
    }
    return g_log;
}

void
ProcessGDBRemoteLog::ListLogCategories (Stream *strm)
{
    strm->Printf("Logging categories for '%s':\n"
        "\tall - turn on all available logging categories\n"
        "\tasync - log asynchronous activity\n"
        "\tbreak - log breakpoints\n"
        "\tcommunication - log communication activity\n"
        "\tdefault - enable the default set of logging categories for liblldb\n"
        "\tpackets - log gdb remote packets\n"
        "\tmemory - log memory reads and writes\n"
        "\tdata-short - log memory bytes for memory reads and writes for short transactions only\n"
        "\tdata-long - log memory bytes for memory reads and writes for all transactions\n"
        "\tprocess - log process events and activities\n"
        "\tthread - log thread events and activities\n"
        "\tstep - log step related activities\n"
        "\tverbose - enable verbose loggging\n"
        "\twatch - log watchpoint related activities\n", ProcessGDBRemote::GetPluginNameStatic());
}


void
ProcessGDBRemoteLog::LogIf (uint32_t mask, const char *format, ...)
{
    Log *log = ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (mask);
    if (log)
    {
        va_list args;
        va_start (args, format);
        log->VAPrintf (format, args);
        va_end (args);
    }
}
