//===-- ProcessMacOSXLog.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProcessMacOSXLog.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Core/StreamFile.h"

#include "ProcessMacOSX.h"

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

LogSP
ProcessMacOSXLog::GetLogIfAllCategoriesSet (uint32_t mask)
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
ProcessMacOSXLog::DisableLog (Args &args, Stream *feedback_strm)
{
    LogSP log (GetLog ());
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
            
                if      (::strcasecmp (arg, "all")        == 0   ) flag_bits &= ~PD_LOG_ALL;
                else if (::strcasestr (arg, "break")      == arg ) flag_bits &= ~PD_LOG_BREAKPOINTS;
                else if (::strcasecmp (arg, "default")    == 0   ) flag_bits &= ~PD_LOG_DEFAULT;
                else if (::strcasestr (arg, "exc")        == arg ) flag_bits &= ~PD_LOG_EXCEPTIONS;
                else if (::strcasecmp (arg, "memory")     == 0   ) flag_bits &= ~PD_LOG_MEMORY;
                else if (::strcasecmp (arg, "data-short") == 0   ) flag_bits &= ~PD_LOG_MEMORY_DATA_SHORT;
                else if (::strcasecmp (arg, "data-long")  == 0   ) flag_bits &= ~PD_LOG_MEMORY_DATA_LONG;
                else if (::strcasecmp (arg, "protections")== 0   ) flag_bits &= ~PD_LOG_PROCESS;
                else if (::strcasecmp (arg, "step")       == 0   ) flag_bits &= ~PD_LOG_STEP;
                else if (::strcasecmp (arg, "task")       == 0   ) flag_bits &= ~PD_LOG_TASK;
                else if (::strcasecmp (arg, "thread")     == 0   ) flag_bits &= ~PD_LOG_THREAD;
                else if (::strcasecmp (arg, "verbose")    == 0   ) flag_bits &= ~PD_LOG_VERBOSE;
                else if (::strcasestr (arg, "watch")      == arg ) flag_bits &= ~PD_LOG_WATCHPOINTS;
                else
                {
                    feedback_strm->Printf("error: unrecognized log category '%s'\n", arg);
                    ListLogCategories (feedback_strm);
                }
            }
        }
        if (flag_bits == 0)
            GetLog().reset();
        else
            log->GetMask().Reset (flag_bits);
    }
}

LogSP
ProcessMacOSXLog::EnableLog (StreamSP &log_stream_sp, uint32_t log_options, Args &args, Stream *feedback_strm)
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
        uint32_t flag_bits = 0;
        bool got_unknown_category = false;
        const size_t argc = args.GetArgumentCount();
        for (size_t i=0; i<argc; ++i)
        {
            const char *arg = args.GetArgumentAtIndex(i);

            if      (::strcasecmp (arg, "all")        == 0   ) flag_bits |= PD_LOG_ALL;
            else if (::strcasestr (arg, "break")      == arg ) flag_bits |= PD_LOG_BREAKPOINTS;
            else if (::strcasecmp (arg, "default")    == 0   ) flag_bits |= PD_LOG_DEFAULT;
            else if (::strcasestr (arg, "exc")        == arg ) flag_bits |= PD_LOG_EXCEPTIONS;
            else if (::strcasecmp (arg, "memory")     == 0   ) flag_bits |= PD_LOG_MEMORY;
            else if (::strcasecmp (arg, "data-short") == 0   ) flag_bits |= PD_LOG_MEMORY_DATA_SHORT;
            else if (::strcasecmp (arg, "data-long")  == 0   ) flag_bits |= PD_LOG_MEMORY_DATA_LONG;
            else if (::strcasecmp (arg, "protections")== 0   ) flag_bits |= PD_LOG_MEMORY_PROTECTIONS;
            else if (::strcasecmp (arg, "process")    == 0   ) flag_bits |= PD_LOG_PROCESS;
            else if (::strcasecmp (arg, "step")       == 0   ) flag_bits |= PD_LOG_STEP;
            else if (::strcasecmp (arg, "task")       == 0   ) flag_bits |= PD_LOG_TASK;
            else if (::strcasecmp (arg, "thread")     == 0   ) flag_bits |= PD_LOG_THREAD;
            else if (::strcasecmp (arg, "verbose")    == 0   ) flag_bits |= PD_LOG_VERBOSE;
            else if (::strcasestr (arg, "watch")      == arg ) flag_bits |= PD_LOG_WATCHPOINTS;
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
            flag_bits = PD_LOG_DEFAULT;
        log->GetMask().Reset(flag_bits);
        log->GetOptions().Reset(log_options);
    }
    return log;
}

void
ProcessMacOSXLog::ListLogCategories (Stream *strm)
{
    strm->Printf("Logging categories for '%s':\n"
        "\tall - turn on all available logging categories\n"
        "\tbreak - log breakpoints\n"
        "\tdefault - enable the default set of logging categories for liblldb\n"
        "\tmemory - log memory reads and writes\n"
        "\tdata-short - log memory bytes for memory reads and writes for short transactions only\n"
        "\tdata-long - log memory bytes for memory reads and writes for all transactions\n"
        "\tprocess - log process events and activities\n"
        "\tprotections - log memory protections\n"
        "\ttask - log mach task calls\n"
        "\tthread - log thread events and activities\n"
        "\tstep - log step related activities\n"
        "\tverbose - enable verbose logging\n"
        "\twatch - log watchpoint related activities\n", ProcessMacOSX::GetPluginNameStatic());
}


void
ProcessMacOSXLog::LogIf (uint32_t mask, const char *format, ...)
{
    LogSP log(ProcessMacOSXLog::GetLogIfAllCategoriesSet (mask));
    if (log)
    {
        va_list args;
        va_start (args, format);
        log->VAPrintf (format, args);
        va_end (args);
    }
}
