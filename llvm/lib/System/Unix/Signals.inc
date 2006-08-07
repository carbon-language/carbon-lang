//===- Signals.cpp - Generic Unix Signals Implementation -----*- C++ -*-===//
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

#include "Unix.h"
#include <vector>
#include <algorithm>
#include <iostream>
#if HAVE_EXECINFO_H
# include <execinfo.h>         // For backtrace().
#endif
#if HAVE_SIGNAL_H
#include <signal.h>
#endif
using namespace llvm;

namespace {

bool StackTraceRequested = false; 

/// InterruptFunction - The function to call if ctrl-c is pressed.
void (*InterruptFunction)() = 0;

std::vector<sys::Path> *FilesToRemove = 0 ;
std::vector<sys::Path> *DirectoriesToRemove = 0;

// IntSigs - Signals that may interrupt the program at any time.
const int IntSigs[] = {
  SIGHUP, SIGINT, SIGQUIT, SIGPIPE, SIGTERM, SIGUSR1, SIGUSR2
};
const int *IntSigsEnd = IntSigs + sizeof(IntSigs)/sizeof(IntSigs[0]);

// KillSigs - Signals that are synchronous with the program that will cause it
// to die.
const int KillSigs[] = {
  SIGILL, SIGTRAP, SIGABRT, SIGFPE, SIGBUS, SIGSEGV, SIGSYS, SIGXCPU, SIGXFSZ
#ifdef SIGEMT
  , SIGEMT
#endif
};
const int *KillSigsEnd = KillSigs + sizeof(KillSigs)/sizeof(KillSigs[0]);

#ifdef HAVE_BACKTRACE
void* StackTrace[256];
#endif

// PrintStackTrace - In the case of a program crash or fault, print out a stack
// trace so that the user has an indication of why and where we died.
//
// On glibc systems we have the 'backtrace' function, which works nicely, but
// doesn't demangle symbols.  In order to backtrace symbols, we fork and exec a
// 'c++filt' process to do the demangling.  This seems like the simplest and
// most robust solution when we can't allocate memory (such as in a signal
// handler).  If we can't find 'c++filt', we fallback to printing mangled names.
//
void PrintStackTrace() {
#ifdef HAVE_BACKTRACE
  // Use backtrace() to output a backtrace on Linux systems with glibc.
  int depth = backtrace(StackTrace, sizeof(StackTrace)/sizeof(StackTrace[0]));
  
  // Create a one-way unix pipe.  The backtracing process writes to PipeFDs[1],
  // the c++filt process reads from PipeFDs[0].
  int PipeFDs[2];
  if (pipe(PipeFDs)) {
    backtrace_symbols_fd(StackTrace, depth, STDERR_FILENO);
    return;
  }

  switch (pid_t ChildPID = fork()) {
  case -1:        // Error forking, print mangled stack trace
    close(PipeFDs[0]);
    close(PipeFDs[1]);
    backtrace_symbols_fd(StackTrace, depth, STDERR_FILENO);
    return;
  default:        // backtracing process
    close(PipeFDs[0]);  // Close the reader side.

    // Print the mangled backtrace into the pipe.
    backtrace_symbols_fd(StackTrace, depth, PipeFDs[1]);
    close(PipeFDs[1]);   // We are done writing.
    while (waitpid(ChildPID, 0, 0) == -1)
      if (errno != EINTR) break;
    return;

  case 0:         // c++filt process
    close(PipeFDs[1]);    // Close the writer side.
    dup2(PipeFDs[0], 0);  // Read from standard input
    close(PipeFDs[0]);    // Close the old descriptor
    dup2(2, 1);           // Revector stdout -> stderr

    // Try to run c++filt or gc++filt.  If neither is found, call back on 'cat'
    // to print the mangled stack trace.  If we can't find cat, just exit.
    execlp("c++filt", "c++filt", (char*)NULL);
    execlp("gc++filt", "gc++filt", (char*)NULL);
    execlp("cat", "cat", (char*)NULL);
    execlp("/bin/cat", "cat", (char*)NULL);
    exit(0);
  }
#endif
}

// SignalHandler - The signal handler that runs...
RETSIGTYPE SignalHandler(int Sig) {
  if (FilesToRemove != 0)
    while (!FilesToRemove->empty()) {
      FilesToRemove->back().eraseFromDisk(true);
      FilesToRemove->pop_back();
    }

  if (DirectoriesToRemove != 0)
    while (!DirectoriesToRemove->empty()) {
      DirectoriesToRemove->back().eraseFromDisk(true);
      DirectoriesToRemove->pop_back();
    }

  if (std::find(IntSigs, IntSigsEnd, Sig) != IntSigsEnd) {
    if (InterruptFunction) {
      void (*IF)() = InterruptFunction;
      InterruptFunction = 0;
      IF();        // run the interrupt function.
      return;
    } else {
      exit(1);   // If this is an interrupt signal, exit the program
    }
  }

  // Otherwise if it is a fault (like SEGV) output the stacktrace to
  // STDERR (if we can) and reissue the signal to die...
  if (StackTraceRequested)
    PrintStackTrace();
  signal(Sig, SIG_DFL);
}

// Just call signal
void RegisterHandler(int Signal) { 
  signal(Signal, SignalHandler); 
}

}


void sys::SetInterruptFunction(void (*IF)()) {
  InterruptFunction = IF;
  RegisterHandler(SIGINT);
}

// RemoveFileOnSignal - The public API
void sys::RemoveFileOnSignal(const sys::Path &Filename) {
  if (FilesToRemove == 0)
    FilesToRemove = new std::vector<sys::Path>;

  FilesToRemove->push_back(Filename);

  std::for_each(IntSigs, IntSigsEnd, RegisterHandler);
  std::for_each(KillSigs, KillSigsEnd, RegisterHandler);
}

// RemoveDirectoryOnSignal - The public API
void sys::RemoveDirectoryOnSignal(const sys::Path& path) {
  // Not a directory?
  sys::FileStatus Status;
  if (path.getFileStatus(Status) || !Status.isDir)
    return;

  if (DirectoriesToRemove == 0)
    DirectoriesToRemove = new std::vector<sys::Path>;

  DirectoriesToRemove->push_back(path);

  std::for_each(IntSigs, IntSigsEnd, RegisterHandler);
  std::for_each(KillSigs, KillSigsEnd, RegisterHandler);
}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIBABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void sys::PrintStackTraceOnErrorSignal() {
  StackTraceRequested = true;
  std::for_each(KillSigs, KillSigsEnd, RegisterHandler);
}
