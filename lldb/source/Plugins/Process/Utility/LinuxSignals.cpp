//===-- LinuxSignals.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <signal.h>

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

    // FIXME we now need *Signals classes on systems that are different OSes (e.g. LinuxSignals
    // needed on MacOSX to debug Linux from MacOSX, and similar scenarios, used by ProcessGDBRemote).  These must be defined
    // not based on OS includes and defines.

#define ADDSIGNAL(S, SUPPRESS, STOP, NOTIFY, DESCRIPTION) \
    AddSignal(SIG ## S, "SIG" #S, #S, SUPPRESS, STOP, NOTIFY, DESCRIPTION)

    ADDSIGNAL(HUP,    false,  true,  true, "hangup");
    ADDSIGNAL(INT,    true,   true,  true, "interrupt");
    ADDSIGNAL(QUIT,   false,  true,  true, "quit");
    ADDSIGNAL(ILL,    false,  true,  true, "illegal instruction");
    ADDSIGNAL(TRAP,   true,   true,  true, "trace trap (not reset when caught)");
    ADDSIGNAL(ABRT,   false,  true,  true, "abort");
    ADDSIGNAL(IOT,    false,  true,  true, "abort");
    ADDSIGNAL(BUS,    false,  true,  true, "bus error");
    ADDSIGNAL(FPE,    false,  true,  true, "floating point exception");
    ADDSIGNAL(KILL,   false,  true,  true, "kill");
    ADDSIGNAL(USR1,   false,  true,  true, "user defined signal 1");
    ADDSIGNAL(SEGV,   false,  true,  true, "segmentation violation");
    ADDSIGNAL(USR2,   false,  true,  true, "user defined signal 2");
    ADDSIGNAL(PIPE,   false,  true,  true, "write to pipe with reading end closed");
    ADDSIGNAL(ALRM,   false,  false, true, "alarm");
    ADDSIGNAL(TERM,   false,  true,  true, "termination requested");
#ifdef SIGSTKFLT
    ADDSIGNAL(STKFLT, false,  true,  true, "stack fault");
#endif
    ADDSIGNAL(CHLD,   false,  false, true, "child process exit");
    ADDSIGNAL(CONT,   false,  true,  true, "process continue");
    ADDSIGNAL(STOP,   true,   true,  true, "process stop");
    ADDSIGNAL(TSTP,   false,  true,  true, "tty stop");
    ADDSIGNAL(TTIN,   false,  true,  true, "background tty read");
    ADDSIGNAL(TTOU,   false,  true,  true, "background tty write");
    ADDSIGNAL(URG,    false,  true,  true, "urgent data on socket");
    ADDSIGNAL(XCPU,   false,  true,  true, "CPU resource exceeded");
    ADDSIGNAL(XFSZ,   false,  true,  true, "file size limit exceeded");
    ADDSIGNAL(VTALRM, false,  true,  true, "virtual alarm");
    ADDSIGNAL(PROF,   false,  true,  true, "profiling alarm");
    ADDSIGNAL(WINCH,  false,  true,  true, "window size change");
#ifdef SIGPOLL
    ADDSIGNAL(POLL,   false,  true,  true, "pollable event");
#endif
    ADDSIGNAL(IO,     false,  true,  true, "input/output ready");
#ifdef SIGPWR
    ADDSIGNAL(PWR,    false,  true,  true, "power failure");
#endif
    ADDSIGNAL(SYS,    false,  true,  true, "invalid system call");

#undef ADDSIGNAL
}
