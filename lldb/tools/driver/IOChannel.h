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

#include <string>
#include <queue>

#include <editline/readline.h>
#include <histedit.h>
#include <pthread.h>
#include <sys/time.h>

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

    IOChannel (FILE *in,
               FILE *out,
               FILE *err,
               Driver *driver = NULL);

    virtual
    ~IOChannel ();

    bool
    Start ();

    bool
    Stop ();

    static void *
    IOReadThread (void *);

    void
    Run ();

    void
    OutWrite (const char *buffer, size_t len);

    void
    ErrWrite (const char *buffer, size_t len);

    bool
    LibeditGetInput (std::string &);

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

    static unsigned char 
    ElCompletionFn (EditLine *e, int ch);

protected:

    bool
    IsGettingCommand () const;

    void
    SetGettingCommand (bool new_value);

    uint64_t
    Nanoseconds (const struct timeval &time_val) const;

    uint64_t
    ElapsedNanoSecondsSinceEnteringElGets ();
    
private:

    pthread_mutex_t m_output_mutex;
    struct timeval m_enter_elgets_time;

    Driver *m_driver;
    lldb::thread_t m_read_thread;
    bool m_read_thread_should_exit;
    FILE *m_out_file;
    FILE *m_err_file;
    std::queue<std::string> m_command_queue;
    const char *m_completion_key;

    EditLine *m_edit_line;
    History *m_history;
    HistEvent m_history_event;
    bool m_getting_command;

    void
    HistorySaveLoad (bool save);

    unsigned char
    HandleCompletion (EditLine *e, int ch);
};

class IOLocker 
{
public:

    IOLocker (pthread_mutex_t &mutex);

    ~IOLocker ();

protected:

    pthread_mutex_t *m_mutex_ptr;

private:

    IOLocker (const IOLocker&);
    const IOLocker& operator= (const IOLocker&);
};

#endif  // lldb_IOChannel_h_
