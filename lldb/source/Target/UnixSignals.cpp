//===-- UnixSignals.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/UnixSignals.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb_private;

UnixSignals::Signal::Signal (const char *name, bool default_suppress, bool default_stop, bool default_notify) :
    m_name (name),
    m_conditions ()
{
    m_conditions[Signal::eCondSuppress] = default_suppress;
    m_conditions[Signal::eCondStop]     = default_stop;
    m_conditions[Signal::eCondNotify]   = default_notify;
}

//----------------------------------------------------------------------
// UnixSignals constructor
//----------------------------------------------------------------------
UnixSignals::UnixSignals ()
{
    Reset ();
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
UnixSignals::~UnixSignals ()
{
}

void
UnixSignals::Reset ()
{
    // This builds one standard set of Unix Signals.  If yours aren't quite in this
    // order, you can either subclass this class, and use Add & Remove to change them
    // or you can subclass and build them afresh in your constructor;
    m_signals.clear();
    //        SIGNO NAME         SUPPRESS   STOP   NOTIFY
    //        ===== ============ =========  ====== ======
    AddSignal(1,    "SIGHUP",    false,     true,  true );    // 1     hangup
    AddSignal(2,    "SIGINT",    true,      true,  true );    // 2     interrupt
    AddSignal(3,    "SIGQUIT",   false,     true,  true );    // 3     quit
    AddSignal(4,    "SIGILL",    false,     true,  true );    // 4     illegal instruction (not reset when caught)
    AddSignal(5,    "SIGTRAP",   true,      true,  true );    // 5     trace trap (not reset when caught)
    AddSignal(6,    "SIGABRT",   false,     true,  true );    // 6     abort()
    AddSignal(7,    "SIGEMT",    false,     true,  true );    // 7     pollable event ([XSR] generated, not supported)
    AddSignal(8,    "SIGFPE",    false,     true,  true );    // 8     floating point exception
    AddSignal(9,    "SIGKILL",   false,     true,  true );    // 9     kill (cannot be caught or ignored)
    AddSignal(10,   "SIGBUS",    false,     true,  true );    // 10    bus error
    AddSignal(11,   "SIGSEGV",   false,     true,  true );    // 11    segmentation violation
    AddSignal(12,   "SIGSYS",    false,     true,  true );    // 12    bad argument to system call
    AddSignal(13,   "SIGPIPE",   false,     true,  true );    // 13    write on a pipe with no one to read it
    AddSignal(14,   "SIGALRM",   false,     false, true );    // 14    alarm clock
    AddSignal(15,   "SIGTERM",   false,     true,  true );    // 15    software termination signal from kill
    AddSignal(16,   "SIGURG",    false,     false, false);    // 16    urgent condition on IO channel
    AddSignal(17,   "SIGSTOP",   false,     true,  true );    // 17    sendable stop signal not from tty
    AddSignal(18,   "SIGTSTP",   false,     true,  true );    // 18    stop signal from tty
    AddSignal(19,   "SIGCONT",   false,     true,  true );    // 19    continue a stopped process
    AddSignal(20,   "SIGCHLD",   false,     false, true );    // 20    to parent on child stop or exit
    AddSignal(21,   "SIGTTIN",   false,     true,  true );    // 21    to readers pgrp upon background tty read
    AddSignal(22,   "SIGTTOU",   false,     true,  true );    // 22    like TTIN for output if (tp->t_local&LTOSTOP)
    AddSignal(23,   "SIGIO",     false,     false, false);    // 23    input/output possible signal
    AddSignal(24,   "SIGXCPU",   false,     true,  true );    // 24    exceeded CPU time limit
    AddSignal(25,   "SIGXFSZ",   false,     true,  true );    // 25    exceeded file size limit
    AddSignal(26,   "SIGVTALRM", false,     false, false);    // 26    virtual time alarm
    AddSignal(27,   "SIGPROF",   false,     false, false);    // 27    profiling time alarm
    AddSignal(28,   "SIGWINCH",  false,     false, false);    // 28    window size changes
    AddSignal(29,   "SIGINFO",   false,     true,  true );    // 29    information request
    AddSignal(30,   "SIGUSR1",   false,     true,  true );    // 30    user defined signal 1
    AddSignal(31,   "SIGUSR2",   false,     true,  true );    // 31    user defined signal 2
}

void
UnixSignals::AddSignal (int signo, const char *name, bool default_suppress, bool default_stop, bool default_notify)
{
    collection::iterator iter = m_signals.find (signo);
    struct Signal new_signal (name, default_suppress, default_stop, default_notify);

    if (iter != m_signals.end())
        m_signals.erase (iter);

    m_signals.insert (iter, collection::value_type (signo, new_signal));
}

void
UnixSignals::RemoveSignal (int signo)
{
    collection::iterator pos = m_signals.find (signo);
    if (pos != m_signals.end())
        m_signals.erase (pos);
}

UnixSignals::Signal *
UnixSignals::GetSignalByName (const char *name, int32_t &signo)
{
    ConstString const_name (name);

    collection::iterator pos, end = m_signals.end ();
    for (pos = m_signals.begin (); pos != end; pos++)
    {
        if (const_name == (*pos).second.m_name)
        {
            signo = (*pos).first;
            return &((*pos).second);
        }
    }
    return NULL;
}


const UnixSignals::Signal *
UnixSignals::GetSignalByName (const char *name, int32_t &signo) const
{
    ConstString const_name (name);

    collection::const_iterator pos, end = m_signals.end ();
    for (pos = m_signals.begin (); pos != end; pos++)
    {
        if (const_name == (*pos).second.m_name)
        {
            signo = (*pos).first;
            return &((*pos).second);
        }
    }
    return NULL;
}

const char *
UnixSignals::GetSignalAsCString (int signo) const
{
    collection::const_iterator pos = m_signals.find (signo);
    if (pos == m_signals.end())
        return NULL;
    else
        return (*pos).second.m_name.GetCString ();
}


bool
UnixSignals::SignalIsValid (int32_t signo) const
{
    return m_signals.find (signo) != m_signals.end();
}


int32_t
UnixSignals::GetSignalNumberFromName (const char *name) const
{
    int32_t signo;
    const Signal *signal = GetSignalByName (name, signo);
    if (signal == NULL)
        return LLDB_INVALID_SIGNAL_NUMBER;
    else
        return signo;
}

int32_t
UnixSignals::GetFirstSignalNumber () const
{
    if (m_signals.empty())
        return LLDB_INVALID_SIGNAL_NUMBER;

    return (*m_signals.begin ()).first;
}

int32_t
UnixSignals::GetNextSignalNumber (int32_t current_signal) const
{
    collection::const_iterator pos = m_signals.find (current_signal);
    collection::const_iterator end = m_signals.end();
    if (pos == end)
        return LLDB_INVALID_SIGNAL_NUMBER;
    else
    {
        pos++;
        if (pos == end)
            return LLDB_INVALID_SIGNAL_NUMBER;
        else
            return (*pos).first;
    }
}

const char *
UnixSignals::GetSignalInfo
(
    int32_t signo,
    bool &should_suppress,
    bool &should_stop,
    bool &should_notify
) const
{
    collection::const_iterator pos = m_signals.find (signo);
    if (pos == m_signals.end())
        return NULL;
    else
    {
        const Signal &signal = (*pos).second;
        should_suppress = signal.m_conditions[Signal::eCondSuppress];
        should_stop     = signal.m_conditions[Signal::eCondStop];
        should_notify   = signal.m_conditions[Signal::eCondNotify];
        return signal.m_name.AsCString("");
    }
}

bool
UnixSignals::GetCondition
(
    int32_t signo,
    UnixSignals::Signal::Condition cond_pos
) const
{
    collection::const_iterator pos = m_signals.find (signo);
    if (pos == m_signals.end())
        return false;
    else
        return (*pos).second.m_conditions[cond_pos];
}

bool
UnixSignals::SetCondition (int32_t signo, UnixSignals::Signal::Condition cond_pos, bool value)
{
    collection::iterator pos = m_signals.find (signo);
    if (pos == m_signals.end())
        return false;
    else
    {
        bool ret_value = (*pos).second.m_conditions[cond_pos];
        (*pos).second.m_conditions[cond_pos] = value;
        return ret_value;
    }
}

bool
UnixSignals::SetCondition (const char *signal_name, UnixSignals::Signal::Condition cond_pos, bool value)
{
    int32_t signo;
    Signal *signal = GetSignalByName (signal_name, signo);
    if (signal == NULL)
        return false;
    else
    {
        bool ret_value = signal->m_conditions[cond_pos];
        signal->m_conditions[cond_pos] = value;
        return ret_value;
    }
}

bool
UnixSignals::GetShouldSuppress (int signo) const
{
    return GetCondition (signo, Signal::eCondSuppress);
}

bool
UnixSignals::SetShouldSuppress (int signo, bool value)
{
    return SetCondition (signo, Signal::eCondSuppress, value);
}

bool
UnixSignals::SetShouldSuppress (const char *signal_name, bool value)
{
    return SetCondition (signal_name, Signal::eCondSuppress, value);
}

bool
UnixSignals::GetShouldStop (int signo) const
{
    return GetCondition (signo, Signal::eCondStop);
}

bool
UnixSignals::SetShouldStop (int signo, bool value)
{
    return SetCondition (signo, Signal::eCondStop, value);
}

bool
UnixSignals::SetShouldStop (const char *signal_name, bool value)
{
    return SetCondition (signal_name, Signal::eCondStop, value);
}

bool
UnixSignals::GetShouldNotify (int signo) const
{
    return GetCondition (signo, Signal::eCondNotify);
}

bool
UnixSignals::SetShouldNotify (int signo, bool value)
{
    return SetCondition (signo, Signal::eCondNotify, value);
}

bool
UnixSignals::SetShouldNotify (const char *signal_name, bool value)
{
    return SetCondition (signal_name, Signal::eCondNotify, value);
}
