//===-- CommandReturnObject.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandReturnObject_h_
#define liblldb_CommandReturnObject_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/STLUtils.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/StreamTee.h"

namespace lldb_private {


class CommandReturnObject
{
public:

    CommandReturnObject ();
    
    ~CommandReturnObject ();

    const char *
    GetOutputData ()
    {
        if (m_output_stream_string_sp)
            return static_cast<StreamString *>(m_output_stream_string_sp.get())->GetData();
        else
            return "";
    }

    const char *
    GetErrorData ()
    {
        if (m_error_stream_string_sp)
            return static_cast<StreamString *>(m_error_stream_string_sp.get())->GetData();
        else
            return "";
    }

    Stream &
    GetOutputStream ()
    {
        if (!m_output_stream_string_sp)
        {
            StreamString *new_stream = new StreamString();
            m_output_stream_string_sp.reset (new_stream);
            m_output_stream.SetStream1 (m_output_stream_string_sp);
        }   
        return m_output_stream;
    }

    Stream &
    GetErrorStream ()
    {
        if (!m_error_stream_string_sp)
        {
            StreamString *new_stream = new StreamString();
            m_error_stream_string_sp.reset (new_stream);
            m_error_stream.SetStream1 (m_error_stream_string_sp);
        }   
        return m_error_stream;
    }

    void
    SetImmediateOutputFile (FILE *fh)
    {
        lldb::StreamSP new_stream_sp (new StreamFile (fh, false));
        m_output_stream.SetStream2 (new_stream_sp);
    }
    
    void
    SetImmediateErrorFile (FILE *fh)
    {
        lldb::StreamSP new_stream_sp (new StreamFile (fh, false));
        SetImmediateOutputStream (new_stream_sp);
    }
    
    void
    SetImmediateOutputStream (lldb::StreamSP &new_stream_sp)
    {
        m_output_stream.SetStream2 (new_stream_sp);
    }
    
    void
    SetImmediateErrorStream (lldb::StreamSP &new_stream_sp)
    {
        m_error_stream.SetStream2 (new_stream_sp);
    }
    
    lldb::StreamSP &
    GetImmediateOutputStream ()
    {
        return m_output_stream.GetStream2 ();
    }
    
    lldb::StreamSP &
    GetImmediateErrorStream ()
    {
        return m_error_stream.GetStream2 ();
    }
    
    void
    Clear();

    void
    AppendMessage (const char *in_string, int len = -1);

    void
    AppendMessageWithFormat (const char *format, ...);

    void
    AppendRawWarning (const char *in_string, int len = -1);

    void
    AppendWarning (const char *in_string, int len = -1);

    void
    AppendWarningWithFormat (const char *format, ...);

    void
    AppendError (const char *in_string, int len = -1);

    void
    AppendRawError (const char *in_string, int len = -1);

    void
    AppendErrorWithFormat (const char *format, ...);

    lldb::ReturnStatus
    GetStatus();

    void
    SetStatus (lldb::ReturnStatus status);

    bool
    Succeeded ();

    bool
    HasResult ();

    bool GetDidChangeProcessState ();

    void SetDidChangeProcessState (bool b);

private:
    lldb::StreamSP m_output_stream_string_sp;
    lldb::StreamSP m_error_stream_string_sp;
    StreamTee    m_output_stream;
    StreamTee    m_error_stream;
    
    lldb::ReturnStatus m_status;
    bool m_did_change_process_state;
};

} // namespace lldb_private

#endif  // liblldb_CommandReturnObject_h_
