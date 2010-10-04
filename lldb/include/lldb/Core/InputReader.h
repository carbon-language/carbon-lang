//===-- InputReader.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_InputReader_h_
#define liblldb_InputReader_h_

#include <termios.h>

#include "lldb/lldb-include.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/Predicate.h"


namespace lldb_private {

class InputReader
{
public:

    typedef size_t (*Callback) (void *baton, 
                                InputReader &reader, 
                                lldb::InputReaderAction notification,
                                const char *bytes, 
                                size_t bytes_len);

    InputReader (Debugger &debugger);

    virtual
    ~InputReader ();

    virtual Error
    Initialize (Callback callback, 
                void *baton,
                lldb::InputReaderGranularity token_size,
                const char *end_token,
                const char *prompt,
                bool echo);
    
    bool
    IsDone () const
    {
        return m_done;
    }

    void
    SetIsDone (bool b)
    {
        m_done = b;
    }

    lldb::InputReaderGranularity
    GetGranularity () const
    {
        return m_granularity;
    }

    bool
    GetEcho () const
    {
        return m_echo;
    }

    // Subclasses _can_ override this function to get input as it comes in
    // without any granularity
    virtual size_t
    HandleRawBytes (const char *bytes, size_t bytes_len);

    Debugger &
    GetDebugger()
    {
        return m_debugger;
    }

    bool 
    IsActive () const
    {
        return m_active;
    }

    const char *
    GetPrompt () const;

    void
    RefreshPrompt();
    
    // If you want to read from an input reader synchronously, then just initialize the
    // reader and then call WaitOnReaderIsDone, which will return when the reader is popped.
    void
    WaitOnReaderIsDone ();

protected:
    friend class Debugger;

    void
    Notify (lldb::InputReaderAction notification);

    Debugger &m_debugger;
    Callback m_callback;
    void *m_callback_baton;
    std::string m_end_token;
    std::string m_prompt;
    lldb::InputReaderGranularity m_granularity;
    bool m_done;
    bool m_echo;
    bool m_active;
    Predicate<bool> m_reader_done;

private:
    DISALLOW_COPY_AND_ASSIGN (InputReader);

};

} // namespace lldb_private

#endif // #ifndef liblldb_InputReader_h_
