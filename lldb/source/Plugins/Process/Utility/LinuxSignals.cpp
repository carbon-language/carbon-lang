//===-- LinuxSignals.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "LinuxSignals.h"

using namespace process_linux;

LinuxSignals::LinuxSignals()
    : UnixSignals()
{
    Reset();
}

void
LinuxSignals::Reset()
{
    m_signals.clear();

    AddSignal (1,    "SIGHUP",    "HUP",     false,   true , true , "hangup");
    AddSignal (2,    "SIGINT",    "INT",     true ,   true , true , "interrupt");
    AddSignal (3,    "SIGQUIT",   "QUIT",    false,   true , true , "quit");
    AddSignal (4,    "SIGILL",    "ILL",     false,   true , true , "illegal instruction");
    AddSignal (5,    "SIGTRAP",   "TRAP",    true ,   true , true , "trace trap (not reset when caught)");
    AddSignal (6,    "SIGABRT",   "ABRT",    false,   true , true , "abort()");
    AddSignal (6,    "SIGIOT",    "IOT",     false,   true , true , "IOT trap");
    AddSignal (7,    "SIGBUS",    "BUS",     false,   true , true , "bus error");
    AddSignal (8,    "SIGFPE",    "FPE",     false,   true , true , "floating point exception");
    AddSignal (9,    "SIGKILL",   "KILL",    false,   true , true , "kill");
    AddSignal (10,   "SIGUSR1",   "USR1",    false,   true , true , "user defined signal 1");
    AddSignal (11,   "SIGSEGV",   "SEGV",    false,   true , true , "segmentation violation");
    AddSignal (12,   "SIGUSR2",   "USR2",    false,   true , true , "user defined signal 2");
    AddSignal (13,   "SIGPIPE",   "PIPE",    false,   true , true , "write to pipe with reading end closed");
    AddSignal (14,   "SIGALRM",   "ALRM",    false,   false, false, "alarm");
    AddSignal (15,   "SIGTERM",   "TERM",    false,   true , true , "termination requested");
    AddSignal (16,   "SIGSTKFLT", "STKFLT",  false,   true , true , "stack fault");
    AddSignal (16,   "SIGCLD",    "CLD",     false,   false, true , "same as SIGCHLD");
    AddSignal (17,   "SIGCHLD",   "CHLD",    false,   false, true , "child status has changed");
    AddSignal (18,   "SIGCONT",   "CONT",    false,   true , true , "process continue");
    AddSignal (19,   "SIGSTOP",   "STOP",    true ,   true , true , "process stop");
    AddSignal (20,   "SIGTSTP",   "TSTP",    false,   true , true , "tty stop");
    AddSignal (21,   "SIGTTIN",   "TTIN",    false,   true , true , "background tty read");
    AddSignal (22,   "SIGTTOU",   "TTOU",    false,   true , true , "background tty write");
    AddSignal (23,   "SIGURG",    "URG",     false,   true , true , "urgent data on socket");
    AddSignal (24,   "SIGXCPU",   "XCPU",    false,   true , true , "CPU resource exceeded");
    AddSignal (25,   "SIGXFSZ",   "XFSZ",    false,   true , true , "file size limit exceeded");
    AddSignal (26,   "SIGVTALRM", "VTALRM",  false,   true , true , "virtual time alarm");
    AddSignal (27,   "SIGPROF",   "PROF",    false,   false, false, "profiling time alarm");
    AddSignal (28,   "SIGWINCH",  "WINCH",   false,   true , true , "window size changes");
    AddSignal (29,   "SIGPOLL",   "POLL",    false,   true , true , "pollable event");
    AddSignal (29,   "SIGIO",     "IO",      false,   true , true , "input/output ready");
    AddSignal (30,   "SIGPWR",    "PWR",     false,   true , true , "power failure");
    AddSignal (31,   "SIGSYS",    "SYS",     false,   true , true , "invalid system call");
}
