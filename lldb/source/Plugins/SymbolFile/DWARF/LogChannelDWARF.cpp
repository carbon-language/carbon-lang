//===-- LogChannelDWARF.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LogChannelDWARF.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "SymbolFileDWARF.h"

using namespace lldb;
using namespace lldb_private;


// when the one and only logging channel is abled, then this will be non NULL.
static LogChannelDWARF* g_log_channel = NULL;

LogChannelDWARF::LogChannelDWARF () :
    LogChannel ()
{
}

LogChannelDWARF::~LogChannelDWARF ()
{
}


void
LogChannelDWARF::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   LogChannelDWARF::CreateInstance);
}

void
LogChannelDWARF::Terminate()
{
    PluginManager::UnregisterPlugin (LogChannelDWARF::CreateInstance);
}

LogChannel*
LogChannelDWARF::CreateInstance ()
{
    return new LogChannelDWARF ();
}

const char *
LogChannelDWARF::GetPluginNameStatic()
{
    return SymbolFileDWARF::GetPluginNameStatic();
}

const char *
LogChannelDWARF::GetPluginDescriptionStatic()
{
    return "DWARF log channel for debugging plug-in issues.";
}

const char *
LogChannelDWARF::GetPluginName()
{
    return GetPluginDescriptionStatic();
}

const char *
LogChannelDWARF::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
LogChannelDWARF::GetPluginVersion()
{
    return 1;
}


void
LogChannelDWARF::Delete ()
{
    g_log_channel = NULL;
    m_log_sp.reset();
}


void
LogChannelDWARF::Disable (Args &categories, Stream *feedback_strm)
{
    if (!m_log_sp)
        return;

    g_log_channel = this;
    uint32_t flag_bits = m_log_sp->GetMask().Get();
    const size_t argc = categories.GetArgumentCount();
    for (size_t i = 0; i < argc; ++i)
    {
         const char *arg = categories.GetArgumentAtIndex(i);

        if      (::strcasecmp (arg, "all")        == 0   ) flag_bits &= ~DWARF_LOG_ALL;
        else if (::strcasecmp (arg, "info")       == 0   ) flag_bits &= ~DWARF_LOG_DEBUG_INFO;
        else if (::strcasecmp (arg, "line")       == 0   ) flag_bits &= ~DWARF_LOG_DEBUG_LINE;
        else if (::strcasecmp (arg, "pubnames")   == 0   ) flag_bits &= ~DWARF_LOG_DEBUG_PUBNAMES;
        else if (::strcasecmp (arg, "pubtypes")   == 0   ) flag_bits &= ~DWARF_LOG_DEBUG_PUBTYPES;
        else if (::strcasecmp (arg, "aranges")    == 0   ) flag_bits &= ~DWARF_LOG_DEBUG_ARANGES;
        else if (::strcasecmp (arg, "default")    == 0   ) flag_bits &= ~DWARF_LOG_DEFAULT;
        else
        {
            feedback_strm->Printf("error: unrecognized log category '%s'\n", arg);
            ListCategories (feedback_strm);
        }
   }
    
    if (flag_bits == 0)
        Delete ();
    else
        m_log_sp->GetMask().Reset (flag_bits);

    return;
}

bool
LogChannelDWARF::Enable
(
    StreamSP &log_stream_sp,
    uint32_t log_options,
    Stream *feedback_strm,  // Feedback stream for argument errors etc
    const Args &categories  // The categories to enable within this logging stream, if empty, enable default set
)
{
    Delete ();

    m_log_sp.reset(new Log (log_stream_sp));
    g_log_channel = this;
    uint32_t flag_bits = 0;
    bool got_unknown_category = false;
    const size_t argc = categories.GetArgumentCount();
    for (size_t i=0; i<argc; ++i)
    {
        const char *arg = categories.GetArgumentAtIndex(i);

        if      (::strcasecmp (arg, "all")        == 0   ) flag_bits |= DWARF_LOG_ALL;
        else if (::strcasecmp (arg, "info")       == 0   ) flag_bits |= DWARF_LOG_DEBUG_INFO;
        else if (::strcasecmp (arg, "line")       == 0   ) flag_bits |= DWARF_LOG_DEBUG_LINE;
        else if (::strcasecmp (arg, "pubnames")   == 0   ) flag_bits |= DWARF_LOG_DEBUG_PUBNAMES;
        else if (::strcasecmp (arg, "pubtypes")   == 0   ) flag_bits |= DWARF_LOG_DEBUG_PUBTYPES;
        else if (::strcasecmp (arg, "aranges")    == 0   ) flag_bits |= DWARF_LOG_DEBUG_ARANGES;
        else if (::strcasecmp (arg, "default")    == 0   ) flag_bits |= DWARF_LOG_DEFAULT;
        else
        {
            feedback_strm->Printf("error: unrecognized log category '%s'\n", arg);
            if (got_unknown_category == false)
            {
                got_unknown_category = true;
                ListCategories (feedback_strm);
            }
        }
    }
    if (flag_bits == 0)
        flag_bits = DWARF_LOG_DEFAULT;
    m_log_sp->GetMask().Reset(flag_bits);
    m_log_sp->GetOptions().Reset(log_options);
    return m_log_sp.get() != NULL;
}

void
LogChannelDWARF::ListCategories (Stream *strm)
{
    strm->Printf ("Logging categories for '%s':\n"
                  "  all - turn on all available logging categories\n"
                  "  info - log the parsing if .debug_info\n"
                  "  line - log the parsing if .debug_line\n"
                  "  pubnames - log the parsing if .debug_pubnames\n"
                  "  pubtypes - log the parsing if .debug_pubtypes\n\n",
                  SymbolFileDWARF::GetPluginNameStatic());
}

Log *
LogChannelDWARF::GetLog ()
{
    if (g_log_channel)
        return g_log_channel->m_log_sp.get();
    else
        return NULL;
}

Log *
LogChannelDWARF::GetLogIfAll (uint32_t mask)
{
    Log *log = GetLog();
    if (log)
    {
        if (log->GetMask().AllSet(mask))
            return log;
    }
    return NULL;
}


void
LogChannelDWARF::LogIf (uint32_t mask, const char *format, ...)
{
    if (g_log_channel)
    {
        LogSP log_sp(g_log_channel->m_log_sp);
        va_list args;
        va_start (args, format);
        log_sp->VAPrintf (format, args);
        va_end (args);
    }
}
