//===- Signals.cpp - Signal Handling support ------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// Unix signals occuring while your program is running.
//
//===----------------------------------------------------------------------===//

#include "Support/Signals.h"
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include "Config/config.h"     // Get the signal handler return type
#ifdef HAVE_EXECINFO_H
# include <execinfo.h>         // For backtrace().
#endif
#include <signal.h>
#include <unistd.h>
using namespace llvm;

static std::vector<std::string> FilesToRemove;

// IntSigs - Signals that may interrupt the program at any time.
static const int IntSigs[] = {
  SIGHUP, SIGINT, SIGQUIT, SIGKILL, SIGPIPE, SIGTERM, SIGUSR1, SIGUSR2
};
static const int *IntSigsEnd = IntSigs + sizeof(IntSigs)/sizeof(IntSigs[0]);

// KillSigs - Signals that are synchronous with the program that will cause it
// to die.
static const int KillSigs[] = {
  SIGILL, SIGTRAP, SIGABRT, SIGFPE, SIGBUS, SIGSEGV, SIGSYS, SIGXCPU, SIGXFSZ
#ifdef SIGEMT
  , SIGEMT
#endif
};
static const int *KillSigsEnd = KillSigs + sizeof(KillSigs)/sizeof(KillSigs[0]);

static void* StackTrace[256];

// SignalHandler - The signal handler that runs...
static RETSIGTYPE SignalHandler(int Sig) {
  while (!FilesToRemove.empty()) {
    std::remove(FilesToRemove.back().c_str());
    FilesToRemove.pop_back();
  }

  if (std::find(IntSigs, IntSigsEnd, Sig) != IntSigsEnd)
    exit(1);   // If this is an interrupt signal, exit the program

  // Otherwise if it is a fault (like SEGV) output the stacktrace to
  // STDERR (if we can) and reissue the signal to die...
#ifdef HAVE_BACKTRACE
  // Use backtrace() to output a backtrace on Linux systems with glibc.
  int depth = backtrace(StackTrace, sizeof(StackTrace)/sizeof(StackTrace[0]));
  backtrace_symbols_fd(StackTrace, depth, STDERR_FILENO);
#endif
  signal(Sig, SIG_DFL);
}

static void RegisterHandler(int Signal) { signal(Signal, SignalHandler); }

// RemoveFileOnSignal - The public API
void llvm::RemoveFileOnSignal(const std::string &Filename) {
  FilesToRemove.push_back(Filename);

  std::for_each(IntSigs, IntSigsEnd, RegisterHandler);
  std::for_each(KillSigs, KillSigsEnd, RegisterHandler);
}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIBABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void llvm::PrintStackTraceOnErrorSignal() {
  std::for_each(KillSigs, KillSigsEnd, RegisterHandler);
}
