//===-- lldb-log.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private-log.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamFile.h"
#include <string.h>

using namespace lldb;
using namespace lldb_private;


static Log *
LogAccessor (bool get, StreamSP *stream_sp_ptr)
{
    static Log* g_log = NULL; // Leak for now as auto_ptr was being cleaned up
                                // by global constructors before other threads
                                // were done with it.
    if (!get)
    {
        if (g_log)
            delete g_log;
        if (stream_sp_ptr)
            g_log = new Log (*stream_sp_ptr);
        else
            g_log = NULL;
    }

    return g_log;

}

uint32_t
lldb_private::GetLogMask ()
{
    Log *log = LogAccessor (true, NULL);
    if (log)
        return log->GetMask().GetAllFlagBits();
    return 0;
}

bool
lldb_private::IsLogVerbose ()
{
    uint32_t mask = GetLogMask();
    return (mask & LIBLLDB_LOG_VERBOSE);
}

Log *
lldb_private::GetLogIfAllCategoriesSet (uint32_t mask)
{
    Log *log = LogAccessor (true, NULL);
    if (log && mask)
    {
        uint32_t log_mask = log->GetMask().GetAllFlagBits();
        if ((log_mask & mask) != mask)
            return NULL;
    }
    return log;
}

void
lldb_private::LogIfAllCategoriesSet (uint32_t mask, const char *format, ...)
{
    Log *log = GetLogIfAllCategoriesSet (mask);
    if (log)
    {
        va_list args;
        va_start (args, format);
        log->VAPrintf (format, args);
        va_end (args);
    }
}

void
lldb_private::LogIfAnyCategoriesSet (uint32_t mask, const char *format, ...)
{
    Log *log = GetLogIfAnyCategoriesSet (mask);
    if (log)
    {
        va_list args;
        va_start (args, format);
        log->VAPrintf (format, args);
        va_end (args);
    }
}

Log *
lldb_private::GetLogIfAnyCategoriesSet (uint32_t mask)
{
    Log *log = LogAccessor (true, NULL);
    if (log && mask && (mask & log->GetMask().GetAllFlagBits()))
        return log;
    return NULL;
}

void
lldb_private::DisableLog ()
{
    LogAccessor (false, NULL);
}


Log *
lldb_private::EnableLog (StreamSP &log_stream_sp, uint32_t log_options, Args &args, Stream *feedback_strm)
{
    // Try see if there already is a log - that way we can reuse its settings.
    // We could reuse the log in toto, but we don't know that the stream is the same.
    uint32_t flag_bits;
    Log* log = LogAccessor (true, NULL);
    if (log)
        flag_bits = log->GetMask().GetAllFlagBits();
    else
        flag_bits = 0;

    // Now make a new log with this stream.
    log = LogAccessor (false, &log_stream_sp);
    if (log)
    {
        bool got_unknown_category = false;
        const size_t argc = args.GetArgumentCount();
        for (size_t i=0; i<argc; ++i)
        {
            const char *arg = args.GetArgumentAtIndex(i);

            if      (strcasecmp(arg, "all")     == 0  ) flag_bits |= LIBLLDB_LOG_ALL;
            else if (strcasestr(arg, "break")   == arg) flag_bits |= LIBLLDB_LOG_BREAKPOINTS;
            else if (strcasecmp(arg, "default") == 0  ) flag_bits |= LIBLLDB_LOG_DEFAULT;
            else if (strcasestr(arg, "event")   == arg) flag_bits |= LIBLLDB_LOG_EVENTS;
            else if (strcasestr(arg, "expr")    == arg) flag_bits |= LIBLLDB_LOG_EXPRESSIONS;
            else if (strcasestr(arg, "object")  == arg) flag_bits |= LIBLLDB_LOG_OBJECT;
            else if (strcasecmp(arg, "process") == 0  ) flag_bits |= LIBLLDB_LOG_PROCESS;
            else if (strcasecmp(arg, "shlib")   == 0  ) flag_bits |= LIBLLDB_LOG_SHLIB;
            else if (strcasecmp(arg, "state")   == 0  ) flag_bits |= LIBLLDB_LOG_STATE;
            else if (strcasecmp(arg, "step")    == 0  ) flag_bits |= LIBLLDB_LOG_STEP;
            else if (strcasecmp(arg, "thread")  == 0  ) flag_bits |= LIBLLDB_LOG_THREAD;
            else if (strcasecmp(arg, "verbose") == 0  ) flag_bits |= LIBLLDB_LOG_VERBOSE;
            else if (strcasestr(arg, "watch")   == arg) flag_bits |= LIBLLDB_LOG_WATCHPOINTS;
            else if (strcasestr(arg, "temp")   == arg)  flag_bits |= LIBLLDB_LOG_TEMPORARY;
            else if (strcasestr(arg, "comm")   == arg)  flag_bits |= LIBLLDB_LOG_COMMUNICATION;
            else if (strcasestr(arg, "conn")   == arg)  flag_bits |= LIBLLDB_LOG_CONNECTION;
            else if (strcasestr(arg, "host")   == arg)  flag_bits |= LIBLLDB_LOG_HOST;
            else
            {
                feedback_strm->Printf("error: unrecognized log category '%s'\n", arg);
                if (got_unknown_category == false)
                {
                    got_unknown_category = true;
                    ListLogCategories (feedback_strm);
                    return log;
                }
            }
        }

        log->GetMask().SetAllFlagBits(flag_bits);
        log->GetOptions().SetAllFlagBits(log_options);
    }
    return log;
}


void
lldb_private::ListLogCategories (Stream *strm)
{
    strm->Printf("Logging categories for 'lldb':\n"
        "\tall - turn on all available logging categories\n"
        "\tdefault - enable the default set of logging categories for liblldb\n"
        "\tbreak - log breakpoints\n"
        "\tevents - log broadcaster, listener and event queue activities\n"
        "\texpr - log expressions\n"
        "\tobject - log object construction/destruction for important objects\n"
        "\tprocess - log process events and activities\n"
        "\tthread - log thread events and activities\n"
        "\tshlib - log shared library related activities\n"
        "\tstate - log private and public process state changes\n"
        "\tstep - log step related activities\n"
        "\tverbose - enable verbose loggging\n"
        "\twatch - log watchpoint related activities\n");
}
