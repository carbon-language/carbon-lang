//===-- GDBServerLog.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//----------------------------------------------------------------------
//
//  GDBServerLog.cpp
//  liblldb
//
//  Created by Greg Clayton on 6/19/09.
//
//
//----------------------------------------------------------------------

#include "GDBServerLog.h"

using namespace lldb;

static Log *
LogAccessor (bool get, Log *log)
{
    static Log* g_log = NULL; // Leak for now as auto_ptr was being cleaned up
                                // by global constructors before other threads
                                // were done with it.
    if (get)
    {
//      // Debug code below for enabling logging by default
//      if (g_log == NULL)
//      {
//          g_log = new Log("/dev/stdout", false);
//          g_log->GetMask().SetAllFlagBits(GS_LOG_ALL);
//          g_log->GetOptions().Set(LLDB_LOG_OPTION_THREADSAFE | LLDB_LOG_OPTION_PREPEND_THREAD_NAME);
//      }
    }
    else
    {
        if (g_log)
            delete g_log;
        g_log = log;
    }

    return g_log;
}

Log *
GDBServerLog::GetLogIfAllCategoriesSet (uint32_t mask)
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
GDBServerLog::SetLog (Log *log)
{
    LogAccessor (false, log);
}


void
GDBServerLog::LogIf (uint32_t mask, const char *format, ...)
{
    Log *log = GDBServerLog::GetLogIfAllCategoriesSet (mask);
    if (log)
    {
        va_list args;
        va_start (args, format);
        log->VAPrintf (format, args);
        va_end (args);
    }
}
