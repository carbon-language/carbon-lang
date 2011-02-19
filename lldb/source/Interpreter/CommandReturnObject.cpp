//===-- CommandReturnObject.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandReturnObject.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/StreamString.h"

using namespace lldb;
using namespace lldb_private;

CommandReturnObject::CommandReturnObject () :
    m_error_stream_string_sp (),
    m_output_stream_string_sp (),
    m_output_stream (),
    m_error_stream (),
    m_status (eReturnStatusStarted),
    m_did_change_process_state (false)
{
}

CommandReturnObject::~CommandReturnObject ()
{
}

void
CommandReturnObject::AppendErrorWithFormat (const char *format, ...)
{
    va_list args;
    va_start (args, format);
    StreamString sstrm;
    sstrm.PrintfVarArg(format, args);
    va_end (args);

    GetErrorStream().Printf("error: %s", sstrm.GetData());
}

void
CommandReturnObject::AppendMessageWithFormat (const char *format, ...)
{
    va_list args;
    va_start (args, format);
    StreamString sstrm;
    sstrm.PrintfVarArg(format, args);
    va_end (args);

    GetOutputStream().Printf("%s", sstrm.GetData());
}

void
CommandReturnObject::AppendWarningWithFormat (const char *format, ...)
{
    va_list args;
    va_start (args, format);
    StreamString sstrm;
    sstrm.PrintfVarArg(format, args);
    va_end (args);

    GetErrorStream().Printf("warning: %s", sstrm.GetData());
}

void
CommandReturnObject::AppendMessage (const char *in_string, int len)
{
    if (len < 0)
        len = ::strlen (in_string);
    GetOutputStream().Printf("%*.*s\n", len, len, in_string);
}

void
CommandReturnObject::AppendWarning (const char *in_string, int len)
{
    if (len < 0)
        len = ::strlen (in_string);
    GetErrorStream().Printf("warning: %*.*s\n", len, len, in_string);
}

// Similar to AppendWarning, but do not prepend 'warning: ' to message, and
// don't append "\n" to the end of it.

void
CommandReturnObject::AppendRawWarning (const char *in_string, int len)
{
    if (len < 0)
        len = ::strlen (in_string);
    GetErrorStream().Printf("%*.*s", len, len, in_string);
}

void
CommandReturnObject::AppendError (const char *in_string, int len)
{
    if (!in_string)
        return;

    if (len < 0)
        len = ::strlen (in_string);
    GetErrorStream().Printf ("error: %*.*s\n", len, len, in_string);
}

// Similar to AppendError, but do not prepend 'Error: ' to message, and
// don't append "\n" to the end of it.

void
CommandReturnObject::AppendRawError (const char *in_string, int len)
{
    if (len < 0)
        len = ::strlen (in_string);
    GetErrorStream().Printf ("%*.*s", len, len, in_string);
}

void
CommandReturnObject::SetStatus (ReturnStatus status)
{
    m_status = status;
}

ReturnStatus
CommandReturnObject::GetStatus ()
{
    return m_status;
}

bool
CommandReturnObject::Succeeded ()
{
    return m_status <= eReturnStatusSuccessContinuingResult;
}

bool
CommandReturnObject::HasResult ()
{
    return (m_status == eReturnStatusSuccessFinishResult ||
            m_status == eReturnStatusSuccessContinuingResult);
}

void
CommandReturnObject::Clear()
{
    if (m_output_stream_string_sp)
        static_cast<StreamString *>(m_output_stream_string_sp.get())->Clear();
    if (m_error_stream_string_sp)
        static_cast<StreamString *>(m_error_stream_string_sp.get())->Clear();
    m_status = eReturnStatusStarted;
}

bool
CommandReturnObject::GetDidChangeProcessState ()
{
    return m_did_change_process_state;
}

void
CommandReturnObject::SetDidChangeProcessState (bool b)
{
    m_did_change_process_state = b;
}

