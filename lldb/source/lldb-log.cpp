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

                if      (0 == ::strcasecmp(arg, "all"))         flag_bits &= ~LIBLLDB_LOG_ALL;
                else if (0 == ::strcasecmp(arg, "api"))         flag_bits &= ~LIBLLDB_LOG_API;
                else if (0 == ::strncasecmp(arg, "break", 5))   flag_bits &= ~LIBLLDB_LOG_BREAKPOINTS;
                else if (0 == ::strcasecmp(arg, "commands"))    flag_bits &= ~LIBLLDB_LOG_COMMANDS;
                else if (0 == ::strcasecmp(arg, "default"))     flag_bits &= ~LIBLLDB_LOG_DEFAULT;
                else if (0 == ::strcasecmp(arg, "dyld"))        flag_bits &= ~LIBLLDB_LOG_DYNAMIC_LOADER;
                else if (0 == ::strncasecmp(arg, "event", 5))   flag_bits &= ~LIBLLDB_LOG_EVENTS;
                else if (0 == ::strncasecmp(arg, "expr", 4))    flag_bits &= ~LIBLLDB_LOG_EXPRESSIONS;
                else if (0 == ::strncasecmp(arg, "object", 6))  flag_bits &= ~LIBLLDB_LOG_OBJECT;
                else if (0 == ::strcasecmp(arg, "process"))     flag_bits &= ~LIBLLDB_LOG_PROCESS;
                else if (0 == ::strcasecmp(arg, "script"))      flag_bits &= ~LIBLLDB_LOG_SCRIPT;
                else if (0 == ::strcasecmp(arg, "state"))       flag_bits &= ~LIBLLDB_LOG_STATE;
                else if (0 == ::strcasecmp(arg, "step"))        flag_bits &= ~LIBLLDB_LOG_STEP;
                else if (0 == ::strcasecmp(arg, "thread"))      flag_bits &= ~LIBLLDB_LOG_THREAD;
                else if (0 == ::strcasecmp(arg, "verbose"))     flag_bits &= ~LIBLLDB_LOG_VERBOSE;
                else if (0 == ::strncasecmp(arg, "watch", 5))   flag_bits &= ~LIBLLDB_LOG_WATCHPOINTS;
                else if (0 == ::strncasecmp(arg, "temp", 4))    flag_bits &= ~LIBLLDB_LOG_TEMPORARY;
                else if (0 == ::strncasecmp(arg, "comm", 4))    flag_bits &= ~LIBLLDB_LOG_COMMUNICATION;
                else if (0 == ::strncasecmp(arg, "conn", 4))    flag_bits &= ~LIBLLDB_LOG_CONNECTION;
                else if (0 == ::strncasecmp(arg, "host", 4))    flag_bits &= ~LIBLLDB_LOG_HOST;
                else if (0 == ::strncasecmp(arg, "unwind", 6))  flag_bits &= ~LIBLLDB_LOG_UNWIND;
                else if (0 == ::strncasecmp(arg, "types", 5))   flag_bits &= ~LIBLLDB_LOG_TYPES;
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

            if      (0 == ::strcasecmp(arg, "all"))         flag_bits |= LIBLLDB_LOG_ALL;
            else if (0 == ::strcasecmp(arg, "api"))         flag_bits |= LIBLLDB_LOG_API;
            else if (0 == ::strncasecmp(arg, "break", 5))   flag_bits |= LIBLLDB_LOG_BREAKPOINTS;
            else if (0 == ::strcasecmp(arg, "commands"))    flag_bits |= LIBLLDB_LOG_COMMANDS;
            else if (0 == ::strcasecmp(arg, "default"))     flag_bits |= LIBLLDB_LOG_DEFAULT;
            else if (0 == ::strcasecmp(arg, "dyld"))        flag_bits |= LIBLLDB_LOG_DYNAMIC_LOADER;
            else if (0 == ::strncasecmp(arg, "event", 5))   flag_bits |= LIBLLDB_LOG_EVENTS;
            else if (0 == ::strncasecmp(arg, "expr", 4))    flag_bits |= LIBLLDB_LOG_EXPRESSIONS;
            else if (0 == ::strncasecmp(arg, "object", 6))  flag_bits |= LIBLLDB_LOG_OBJECT;
            else if (0 == ::strcasecmp(arg, "process"))     flag_bits |= LIBLLDB_LOG_PROCESS;
            else if (0 == ::strcasecmp(arg, "script"))      flag_bits |= LIBLLDB_LOG_SCRIPT;
            else if (0 == ::strcasecmp(arg, "state"))       flag_bits |= LIBLLDB_LOG_STATE;
            else if (0 == ::strcasecmp(arg, "step"))        flag_bits |= LIBLLDB_LOG_STEP;
            else if (0 == ::strcasecmp(arg, "thread"))      flag_bits |= LIBLLDB_LOG_THREAD;
            else if (0 == ::strcasecmp(arg, "verbose"))     flag_bits |= LIBLLDB_LOG_VERBOSE;
            else if (0 == ::strncasecmp(arg, "watch", 5))   flag_bits |= LIBLLDB_LOG_WATCHPOINTS;
            else if (0 == ::strncasecmp(arg, "temp", 4))    flag_bits |= LIBLLDB_LOG_TEMPORARY;
            else if (0 == ::strncasecmp(arg, "comm", 4))    flag_bits |= LIBLLDB_LOG_COMMUNICATION;
            else if (0 == ::strncasecmp(arg, "conn", 4))    flag_bits |= LIBLLDB_LOG_CONNECTION;
            else if (0 == ::strncasecmp(arg, "host", 4))    flag_bits |= LIBLLDB_LOG_HOST;
            else if (0 == ::strncasecmp(arg, "unwind", 6))  flag_bits |= LIBLLDB_LOG_UNWIND;
            else if (0 == ::strncasecmp(arg, "types", 5))   flag_bits |= LIBLLDB_LOG_TYPES;
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
        "\tcommand - log command argument parsing\n"
        "\tdefault - enable the default set of logging categories for liblldb\n"
        "\tbreak - log breakpoints\n"
        "\tevents - log broadcaster, listener and event queue activities\n"
        "\texpr - log expressions\n"
        "\tobject - log object construction/destruction for important objects\n"
        "\tprocess - log process events and activities\n"
        "\tthread - log thread events and activities\n"
        "\tscript - log events about the script interpreter\n"
        "\tdyld - log shared library related activities\n"
        "\tstate - log private and public process state changes\n"
        "\tstep - log step related activities\n"
        "\tunwind - log stack unwind activities\n"
        "\tverbose - enable verbose logging\n"
        "\twatch - log watchpoint related activities\n"
        "\ttypes - log type system related activities\n");
}
