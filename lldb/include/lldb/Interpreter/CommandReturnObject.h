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
#include "lldb/Core/StreamString.h"

namespace lldb_private {


class CommandReturnObject
{
public:

    CommandReturnObject ();

    ~CommandReturnObject ();

    StreamString &
    GetOutputStream ();

    StreamString &
    GetErrorStream ();

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
    StreamString m_output_stream;
    StreamString m_error_stream;
    lldb::ReturnStatus m_status;
    bool m_did_change_process_state;
};

} // namespace lldb_private

#endif  // liblldb_CommandReturnObject_h_
