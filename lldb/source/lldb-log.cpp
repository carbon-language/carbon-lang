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


// We want to avoid global constructors where code needs to be run so here we
// control access to our static g_log_sp by hiding it in a singleton function
// that will construct the static g_lob_sp the first time this function is 
// called.
static LogSP &
GetLog ()
{
    static LogSP g_log_sp;
    return g_log_sp;
}

uint32_t
lldb_private::GetLogMask ()
{
    LogSP log(GetLog ());
    if (log)
        return log->GetMask().Get();
    return 0;
}

bool
lldb_private::IsLogVerbose ()
{
    uint32_t mask = GetLogMask();
    return (mask & LIBLLDB_LOG_VERBOSE);
}

LogSP
lldb_private::GetLogIfAllCategoriesSet (uint32_t mask)
{
    LogSP log(GetLog ());
    if (log && mask)
    {
        uint32_t log_mask = log->GetMask().Get();
        if ((log_mask & mask) != mask)
            return LogSP();
    }
    return log;
}

void
lldb_private::LogIfAllCategoriesSet (uint32_t mask, const char *format, ...)
{
    LogSP log(GetLogIfAllCategoriesSet (mask));
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
    LogSP log(GetLogIfAnyCategoriesSet (mask));
    if (log)
    {
        va_list args;
        va_start (args, format);
        log->VAPrintf (format, args);
        va_end (args);
    }
}

LogSP
lldb_private::GetLogIfAnyCategoriesSet (uint32_t mask)
{
    LogSP log(GetLog ());
    if (log && mask && (mask & log->GetMask().Get()))
        return log;
    return LogSP();
}

void
lldb_private::DisableLog (Args &args, Stream *feedback_strm)
{
    LogSP log(GetLog ());

    if (log)
    {
        uint32_t flag_bits = 0;
        const size_t argc = args.GetArgumentCount ();
        if (argc > 0)
        {
            flag_bits = log->GetMask().Get();
            for (size_t i = 0; i < argc; ++i)
            {
                const char *arg = args.GetArgumentAtIndex (i);

                if      (strcasecmp(arg, "all")     == 0  ) flag_bits &= ~LIBLLDB_LOG_ALL;
                else if (strcasecmp(arg, "api")     == 0)   flag_bits &= ~LIBLLDB_LOG_API;
                else if (strcasestr(arg, "break")   == arg) flag_bits &= ~LIBLLDB_LOG_BREAKPOINTS;
                else if (strcasecmp(arg, "default") == 0  ) flag_bits &= ~LIBLLDB_LOG_DEFAULT;
                else if (strcasecmp(arg, "dyld")    == 0  ) flag_bits &= ~LIBLLDB_LOG_DYNAMIC_LOADER;
                else if (strcasestr(arg, "event")   == arg) flag_bits &= ~LIBLLDB_LOG_EVENTS;
                else if (strcasestr(arg, "expr")    == arg) flag_bits &= ~LIBLLDB_LOG_EXPRESSIONS;
                else if (strcasestr(arg, "object")  == arg) flag_bits &= ~LIBLLDB_LOG_OBJECT;
                else if (strcasecmp(arg, "process") == 0  ) flag_bits &= ~LIBLLDB_LOG_PROCESS;
                else if (strcasecmp(arg, "script") == 0)    flag_bits &= ~LIBLLDB_LOG_SCRIPT;
                else if (strcasecmp(arg, "state")   == 0  ) flag_bits &= ~LIBLLDB_LOG_STATE;
                else if (strcasecmp(arg, "step")    == 0  ) flag_bits &= ~LIBLLDB_LOG_STEP;
                else if (strcasecmp(arg, "thread")  == 0  ) flag_bits &= ~LIBLLDB_LOG_THREAD;
                else if (strcasecmp(arg, "verbose") == 0  ) flag_bits &= ~LIBLLDB_LOG_VERBOSE;
                else if (strcasestr(arg, "watch")   == arg) flag_bits &= ~LIBLLDB_LOG_WATCHPOINTS;
                else if (strcasestr(arg, "temp")   == arg)  flag_bits &= ~LIBLLDB_LOG_TEMPORARY;
                else if (strcasestr(arg, "comm")   == arg)  flag_bits &= ~LIBLLDB_LOG_COMMUNICATION;
                else if (strcasestr(arg, "conn")   == arg)  flag_bits &= ~LIBLLDB_LOG_CONNECTION;
                else if (strcasestr(arg, "host")   == arg)  flag_bits &= ~LIBLLDB_LOG_HOST;
                else if (strcasestr(arg, "unwind") == arg)  flag_bits &= ~LIBLLDB_LOG_UNWIND;
                else
                {
                    feedback_strm->Printf ("error:  unrecognized log category '%s'\n", arg);
                    ListLogCategories (feedback_strm);
                    return;
                }
                
            }
        }
        if (flag_bits == 0)
            GetLog ().reset();
        else
            log->GetMask().Reset (flag_bits);
    }

    return;
}

LogSP
lldb_private::EnableLog (StreamSP &log_stream_sp, uint32_t log_options, Args &args, Stream *feedback_strm)
{
    // Try see if there already is a log - that way we can reuse its settings.
    // We could reuse the log in toto, but we don't know that the stream is the same.
    uint32_t flag_bits;
    LogSP log(GetLog ());
    if (log)
        flag_bits = log->GetMask().Get();
    else
        flag_bits = 0;

    // Now make a new log with this stream if one was provided
    if (log_stream_sp)
    {
        log = make_shared<Log>(log_stream_sp);
        GetLog () = log;
    }

    if (log)
    {
        bool got_unknown_category = false;
        const size_t argc = args.GetArgumentCount();
        for (size_t i=0; i<argc; ++i)
        {
            const char *arg = args.GetArgumentAtIndex(i);

            if      (strcasecmp(arg, "all")     == 0  ) flag_bits |= LIBLLDB_LOG_ALL;
            else if (strcasecmp(arg, "api")     == 0)   flag_bits |= LIBLLDB_LOG_API;
            else if (strcasestr(arg, "break")   == arg) flag_bits |= LIBLLDB_LOG_BREAKPOINTS;
            else if (strcasecmp(arg, "default") == 0  ) flag_bits |= LIBLLDB_LOG_DEFAULT;
            else if (strcasecmp(arg, "dyld")    == 0  ) flag_bits |= LIBLLDB_LOG_DYNAMIC_LOADER;
            else if (strcasestr(arg, "event")   == arg) flag_bits |= LIBLLDB_LOG_EVENTS;
            else if (strcasestr(arg, "expr")    == arg) flag_bits |= LIBLLDB_LOG_EXPRESSIONS;
            else if (strcasestr(arg, "object")  == arg) flag_bits |= LIBLLDB_LOG_OBJECT;
            else if (strcasecmp(arg, "process") == 0  ) flag_bits |= LIBLLDB_LOG_PROCESS;
            else if (strcasecmp(arg, "script") == 0)    flag_bits |= LIBLLDB_LOG_SCRIPT;
            else if (strcasecmp(arg, "state")   == 0  ) flag_bits |= LIBLLDB_LOG_STATE;
            else if (strcasecmp(arg, "step")    == 0  ) flag_bits |= LIBLLDB_LOG_STEP;
            else if (strcasecmp(arg, "thread")  == 0  ) flag_bits |= LIBLLDB_LOG_THREAD;
            else if (strcasecmp(arg, "verbose") == 0  ) flag_bits |= LIBLLDB_LOG_VERBOSE;
            else if (strcasestr(arg, "watch")   == arg) flag_bits |= LIBLLDB_LOG_WATCHPOINTS;
            else if (strcasestr(arg, "temp")   == arg)  flag_bits |= LIBLLDB_LOG_TEMPORARY;
            else if (strcasestr(arg, "comm")   == arg)  flag_bits |= LIBLLDB_LOG_COMMUNICATION;
            else if (strcasestr(arg, "conn")   == arg)  flag_bits |= LIBLLDB_LOG_CONNECTION;
            else if (strcasestr(arg, "host")   == arg)  flag_bits |= LIBLLDB_LOG_HOST;
            else if (strcasestr(arg, "unwind") == arg)  flag_bits |= LIBLLDB_LOG_UNWIND;
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

        log->GetMask().Reset(flag_bits);
        log->GetOptions().Reset(log_options);
    }
    return log;
}


void
lldb_private::ListLogCategories (Stream *strm)
{
    strm->Printf("Logging categories for 'lldb':\n"
        "\tall - turn on all available logging categories\n"
        "\tapi - enable logging of API calls and return values\n"
        "\tdefault - enable the default set of logging categories for liblldb\n"
        "\tbreak - log breakpoints\n"
        "\tevents - log broadcaster, listener and event queue activities\n"
        "\texpr - log expressions\n"
        "\tobject - log object construction/destruction for important objects\n"
        "\tprocess - log process events and activities\n"
        "\tthread - log thread events and activities\n"
        "\tscript - log events about the script interpreter\n"
        "\tshlib - log shared library related activities\n"
        "\tstate - log private and public process state changes\n"
        "\tstep - log step related activities\n"
        "\tunwind - log stack unwind activities\n"
        "\tverbose - enable verbose loggging\n"
        "\twatch - log watchpoint related activities\n");
}
