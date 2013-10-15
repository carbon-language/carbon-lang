//===-- IOChannel.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_IOChannel_h_
#define lldb_IOChannel_h_

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <string>
#include <queue>
#include "Driver.h"

class IOChannel : public lldb::SBBroadcaster
{
public:
    enum {
        eBroadcastBitHasUserInput     = (1 << 0),
        eBroadcastBitUserInterrupt    = (1 << 1),
        eBroadcastBitThreadShouldExit = (1 << 2),
        eBroadcastBitThreadDidExit    = (1 << 3),
        eBroadcastBitThreadDidStart   = (1 << 4),
        eBroadcastBitsSTDOUT          = (1 << 5),
        eBroadcastBitsSTDERR          = (1 << 6),
        eBroadcastBitsSTDIN           = (1 << 7),
        eAllEventBits                 = 0xffffffff
    };
    
    enum LibeditGetInputResult
    {
        eLibeditGetInputEOF = 0,
        eLibeditGetInputValid = 1,
        eLibeditGetInputEmpty = 2,
        eLibeditGetInputResultError = 4,
        eLibeditGetInputResultUnknown = 0xffffffff
    };

    IOChannel (FILE *editline_in,
               FILE *editline_out,
               FILE *out,
               FILE *err,
               Driver *driver = NULL);

    virtual
    ~IOChannel ();

    bool
    Start ();

    bool
    Stop ();

    static lldb::thread_result_t
    IOReadThread (void *);

    void
    Run ();

    void
    OutWrite (const char *buffer, size_t len, bool asynchronous);

    void
    ErrWrite (const char *buffer, size_t len, bool asynchronous);

    LibeditGetInputResult
    LibeditGetInput (std::string &);
    
    static void
    LibeditOutputBytesReceived (void *baton, const void *src,size_t src_len);

    void
    SetPrompt ();

    void
    RefreshPrompt ();

    void
    AddCommandToQueue (const char *command);

    bool
    GetCommandFromQueue (std::string &cmd);

    int
    CommandQueueSize () const;

    void
    ClearCommandQueue ();

    bool
    CommandQueueIsEmpty () const;

    const char *
    GetPrompt ();

    bool
    EditLineHasCharacters ();
    
    void
    EraseCharsBeforeCursor ();

    static unsigned char 
    ElCompletionFn (EditLine *e, int ch);
    
    void
    ElResize();

protected:

    bool
    IsGettingCommand () const;

    void
    SetGettingCommand (bool new_value);

private:

    std::recursive_mutex m_output_mutex;
    std::condition_variable_any m_output_cond;
    struct timeval m_enter_elgets_time;

    Driver *m_driver;
    lldb::thread_t m_read_thread;
    bool m_read_thread_should_exit;
    FILE *m_out_file;
    FILE *m_err_file;
    FILE *m_editline_out;
    std::queue<std::string> m_command_queue;
    const char *m_completion_key;

    EditLine *m_edit_line;
    History *m_history;
    HistEvent m_history_event;
    bool m_getting_command;
    bool m_expecting_prompt;
    bool m_output_flushed;
    std::string m_prompt_str;  // for accumlating the prompt as it gets written out by editline
    bool m_refresh_request_pending;

    void
    HistorySaveLoad (bool save);

    unsigned char
    HandleCompletion (EditLine *e, int ch);
};

#endif  // lldb_IOChannel_h_
