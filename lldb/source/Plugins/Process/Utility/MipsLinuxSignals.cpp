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

    AddSignal (1,    "SIGHUP",      "HUP",      false,   true , true , "hangup");
    AddSignal (2,    "SIGINT",      "INT",      true ,   true , true , "interrupt");
    AddSignal (3,    "SIGQUIT",     "QUIT",     false,   true , true , "quit");
    AddSignal (4,    "SIGILL",      "ILL",      false,   true , true , "illegal instruction");
    AddSignal (5,    "SIGTRAP",     "TRAP",     true ,   true , true , "trace trap (not reset when caught)");
    AddSignal (6,    "SIGABRT",     "ABRT",     false,   true , true , "abort()");
    AddSignal (6,    "SIGIOT",      "IOT",      false,   true , true , "IOT trap");
    AddSignal (7,    "SIGEMT",      "EMT",      false,   true , true , "terminate process with core dump");
    AddSignal (8,    "SIGFPE",      "FPE",      false,   true , true , "floating point exception");
    AddSignal (9,    "SIGKILL",     "KILL",     false,   true , true , "kill");
    AddSignal (10,   "SIGBUS",      "BUS",      false,   true , true , "bus error");
    AddSignal (11,   "SIGSEGV",     "SEGV",     false,   true , true , "segmentation violation");
    AddSignal (12,   "SIGSYS",      "SYS",      false,   true , true , "invalid system call");
    AddSignal (13,   "SIGPIPE",     "PIPE",     false,   true , true , "write to pipe with reading end closed");
    AddSignal (14,   "SIGALRM",     "ALRM",     false,   false, false, "alarm");
    AddSignal (15,   "SIGTERM",     "TERM",     false,   true , true , "termination requested");
    AddSignal (16,   "SIGUSR1",     "USR1",     false,   true , true , "user defined signal 1");
    AddSignal (17,   "SIGUSR2",     "USR2",     false,   true , true , "user defined signal 2");
    AddSignal (18,   "SIGCLD",      "CLD",      false,   false, true , "same as SIGCHLD");
    AddSignal (18,   "SIGCHLD",     "CHLD",     false,   false, true , "child status has changed");
    AddSignal (19,   "SIGPWR",      "PWR",      false,   true , true , "power failure");
    AddSignal (20,   "SIGWINCH",    "WINCH",    false,   true , true , "window size changes");
    AddSignal (21,   "SIGURG",      "URG",      false,   true , true , "urgent data on socket");
    AddSignal (22,   "SIGIO",       "IO",       false,   true , true , "input/output ready");
    AddSignal (22,   "SIGPOLL",     "POLL",     false,   true , true , "pollable event");
    AddSignal (23,   "SIGSTOP",     "STOP",     true ,   true , true , "process stop");
    AddSignal (24,   "SIGTSTP",     "TSTP",     false,   true , true , "tty stop");
    AddSignal (25,   "SIGCONT",     "CONT",     false,   true , true , "process continue");
    AddSignal (26,   "SIGTTIN",     "TTIN",     false,   true , true , "background tty read");
    AddSignal (27,   "SIGTTOU",     "TTOU",     false,   true , true , "background tty write");
    AddSignal (28,   "SIGVTALRM",   "VTALRM",   false,   true , true , "virtual time alarm");
    AddSignal (29,   "SIGPROF",     "PROF",     false,   false, false, "profiling time alarm");
    AddSignal (30,   "SIGXCPU",     "XCPU",     false,   true , true , "CPU resource exceeded");
    AddSignal (31,   "SIGXFSZ",     "XFSZ",     false,   true , true , "file size limit exceeded");
    AddSignal (32,   "SIG32",       "SIG32",    false,   true , true , "threading library internal signal 1");
    AddSignal (33,   "SIG33",       "SIG33",    false,   true , true , "threading library internal signal 2");
    AddSignal (34,   "SIGRTMIN",    "RTMIN",    false,   true , true , "real time signal 0");
    AddSignal (35,   "SIGRTMIN+1",  "RTMIN+1",  false,   true , true , "real time signal 1");
    AddSignal (36,   "SIGRTMIN+2",  "RTMIN+2",  false,   true , true , "real time signal 2");
    AddSignal (37,   "SIGRTMIN+3",  "RTMIN+3",  false,   true , true , "real time signal 3");
    AddSignal (38,   "SIGRTMIN+4",  "RTMIN+4",  false,   true , true , "real time signal 4");
    AddSignal (39,   "SIGRTMIN+5",  "RTMIN+5",  false,   true , true , "real time signal 5");
    AddSignal (40,   "SIGRTMIN+6",  "RTMIN+6",  false,   true , true , "real time signal 6");
    AddSignal (41,   "SIGRTMIN+7",  "RTMIN+7",  false,   true , true , "real time signal 7");
    AddSignal (42,   "SIGRTMIN+8",  "RTMIN+8",  false,   true , true , "real time signal 8");
    AddSignal (43,   "SIGRTMIN+9",  "RTMIN+9",  false,   true , true , "real time signal 9");
    AddSignal (44,   "SIGRTMIN+10", "RTMIN+10", false,   true , true , "real time signal 10");
    AddSignal (45,   "SIGRTMIN+11", "RTMIN+11", false,   true , true , "real time signal 11");
    AddSignal (46,   "SIGRTMIN+12", "RTMIN+12", false,   true , true , "real time signal 12");
    AddSignal (47,   "SIGRTMIN+13", "RTMIN+13", false,   true , true , "real time signal 13");
    AddSignal (48,   "SIGRTMIN+14", "RTMIN+14", false,   true , true , "real time signal 14");
    AddSignal (49,   "SIGRTMIN+15", "RTMIN+15", false,   true , true , "real time signal 15");
    AddSignal (50,   "SIGRTMAX-14", "RTMAX-14", false,   true , true , "real time signal 16"); // switching to SIGRTMAX-xxx to match "kill -l" output
    AddSignal (51,   "SIGRTMAX-13", "RTMAX-13", false,   true , true , "real time signal 17");
    AddSignal (52,   "SIGRTMAX-12", "RTMAX-12", false,   true , true , "real time signal 18");
    AddSignal (53,   "SIGRTMAX-11", "RTMAX-11", false,   true , true , "real time signal 19");
    AddSignal (54,   "SIGRTMAX-10", "RTMAX-10", false,   true , true , "real time signal 20");
    AddSignal (55,   "SIGRTMAX-9",  "RTMAX-9",  false,   true , true , "real time signal 21");
    AddSignal (56,   "SIGRTMAX-8",  "RTMAX-8",  false,   true , true , "real time signal 22");
    AddSignal (57,   "SIGRTMAX-7",  "RTMAX-7",  false,   true , true , "real time signal 23");
    AddSignal (58,   "SIGRTMAX-6",  "RTMAX-6",  false,   true , true , "real time signal 24");
    AddSignal (59,   "SIGRTMAX-5",  "RTMAX-5",  false,   true , true , "real time signal 25");
    AddSignal (60,   "SIGRTMAX-4",  "RTMAX-4",  false,   true , true , "real time signal 26");
    AddSignal (61,   "SIGRTMAX-3",  "RTMAX-3",  false,   true , true , "real time signal 27");
    AddSignal (62,   "SIGRTMAX-2",  "RTMAX-2",  false,   true , true , "real time signal 28");
    AddSignal (63,   "SIGRTMAX-1",  "RTMAX-1",  false,   true , true , "real time signal 29");
    AddSignal (64,   "SIGRTMAX",    "RTMAX",    false,   true , true , "real time signal 30");
}
