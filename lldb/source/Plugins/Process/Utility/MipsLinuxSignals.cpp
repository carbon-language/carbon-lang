//===-- MipsLinuxSignals.cpp ----------------------------------------*- C++ -*-===//
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
#include "MipsLinuxSignals.h"

using namespace lldb_private::process_linux;

MipsLinuxSignals::MipsLinuxSignals()
    : UnixSignals()
{
    Reset();
}

void
MipsLinuxSignals::Reset()
{
    m_signals.clear();

    AddSignal (1,    "SIGHUP",    "HUP",     false,   true , true , "hangup");
    AddSignal (2,    "SIGINT",    "INT",     true ,   true , true , "interrupt");
    AddSignal (3,    "SIGQUIT",   "QUIT",    false,   true , true , "quit");
    AddSignal (4,    "SIGILL",    "ILL",     false,   true , true , "illegal instruction");
    AddSignal (5,    "SIGTRAP",   "TRAP",    true ,   true , true , "trace trap (not reset when caught)");
    AddSignal (6,    "SIGABRT",   "ABRT",    false,   true , true , "abort()");
    AddSignal (6,    "SIGIOT",    "IOT",     false,   true , true , "IOT trap");
    AddSignal (7,    "SIGEMT",    "EMT",     false,   true , true , "terminate process with core dump");
    AddSignal (8,    "SIGFPE",    "FPE",     false,   true , true , "floating point exception");
    AddSignal (9,    "SIGKILL",   "KILL",    false,   true , true , "kill");
    AddSignal (10,   "SIGBUS",    "BUS",     false,   true , true , "bus error");
    AddSignal (11,   "SIGSEGV",   "SEGV",    false,   true , true , "segmentation violation");
    AddSignal (12,   "SIGSYS",    "SYS",     false,   true , true , "invalid system call");
    AddSignal (13,   "SIGPIPE",   "PIPE",    false,   true , true , "write to pipe with reading end closed");
    AddSignal (14,   "SIGALRM",   "ALRM",    false,   false, false, "alarm");
    AddSignal (15,   "SIGTERM",   "TERM",    false,   true , true , "termination requested");
    AddSignal (16,   "SIGUSR1",   "USR1",    false,   true , true , "user defined signal 1");
    AddSignal (17,   "SIGUSR2",   "USR2",    false,   true , true , "user defined signal 2");
    AddSignal (18,   "SIGCLD",    "CLD",     false,   false, true , "same as SIGCHLD");
    AddSignal (18,   "SIGCHLD",   "CHLD",    false,   false, true , "child status has changed");
    AddSignal (19,   "SIGPWR",    "PWR",     false,   true , true , "power failure");
    AddSignal (20,   "SIGWINCH",  "WINCH",   false,   true , true , "window size changes");
    AddSignal (21,   "SIGURG",    "URG",     false,   true , true , "urgent data on socket");
    AddSignal (22,   "SIGIO",     "IO",      false,   true , true , "input/output ready");
    AddSignal (22,   "SIGPOLL",   "POLL",    false,   true , true , "pollable event");
    AddSignal (23,   "SIGSTOP",   "STOP",    true ,   true , true , "process stop");
    AddSignal (24,   "SIGTSTP",   "TSTP",    false,   true , true , "tty stop");
    AddSignal (25,   "SIGCONT",   "CONT",    false,   true , true , "process continue");
    AddSignal (26,   "SIGTTIN",   "TTIN",    false,   true , true , "background tty read");
    AddSignal (27,   "SIGTTOU",   "TTOU",    false,   true , true , "background tty write");
    AddSignal (28,   "SIGVTALRM", "VTALRM",  false,   true , true , "virtual time alarm");
    AddSignal (29,   "SIGPROF",   "PROF",    false,   false, false, "profiling time alarm");
    AddSignal (30,   "SIGXCPU",   "XCPU",    false,   true , true , "CPU resource exceeded");
    AddSignal (31,   "SIGXFSZ",   "XFSZ",    false,   true , true , "file size limit exceeded");
}
