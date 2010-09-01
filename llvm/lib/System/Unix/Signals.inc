//===- Signals.cpp - Generic Unix Signals Implementation -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// Unix signals occuring while your program is running.
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/System/Mutex.h"
#include <vector>
#include <algorithm>
#if HAVE_EXECINFO_H
# include <execinfo.h>         // For backtrace().
#endif
#if HAVE_SIGNAL_H
#include <signal.h>
#endif
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_DLFCN_H && __GNUG__
#include <dlfcn.h>
#include <cxxabi.h> 
#endif
using namespace llvm;

static RETSIGTYPE SignalHandler(int Sig);  // defined below.

static SmartMutex<true> SignalsMutex;

/// InterruptFunction - The function to call if ctrl-c is pressed.
static void (*InterruptFunction)() = 0;

static std::vector<sys::Path> FilesToRemove;
static std::vector<std::pair<void(*)(void*), void*> > CallBacksToRun;

// IntSigs - Signals that may interrupt the program at any time.
static const int IntSigs[] = {
  SIGHUP, SIGINT, SIGQUIT, SIGPIPE, SIGTERM, SIGUSR1, SIGUSR2
};
static const int *const IntSigsEnd =
  IntSigs + sizeof(IntSigs) / sizeof(IntSigs[0]);

// KillSigs - Signals that are synchronous with the program that will cause it
// to die.
static const int KillSigs[] = {
  SIGILL, SIGTRAP, SIGABRT, SIGFPE, SIGBUS, SIGSEGV
#ifdef SIGSYS
  , SIGSYS
#endif
#ifdef SIGXCPU
  , SIGXCPU
#endif
#ifdef SIGXFSZ
  , SIGXFSZ
#endif
#ifdef SIGEMT
  , SIGEMT
#endif
};
static const int *const KillSigsEnd =
  KillSigs + sizeof(KillSigs) / sizeof(KillSigs[0]);

static unsigned NumRegisteredSignals = 0;
static struct {
  struct sigaction SA;
  int SigNo;
} RegisteredSignalInfo[(sizeof(IntSigs)+sizeof(KillSigs))/sizeof(KillSigs[0])];


static void RegisterHandler(int Signal) {
  assert(NumRegisteredSignals <
         sizeof(RegisteredSignalInfo)/sizeof(RegisteredSignalInfo[0]) &&
         "Out of space for signal handlers!");

  struct sigaction NewHandler;
  
  NewHandler.sa_handler = SignalHandler;
  NewHandler.sa_flags = SA_NODEFER|SA_RESETHAND;
  sigemptyset(&NewHandler.sa_mask); 
  
  // Install the new handler, save the old one in RegisteredSignalInfo.
  sigaction(Signal, &NewHandler,
            &RegisteredSignalInfo[NumRegisteredSignals].SA);
  RegisteredSignalInfo[NumRegisteredSignals].SigNo = Signal;
  ++NumRegisteredSignals;
}

static void RegisterHandlers() {
  // If the handlers are already registered, we're done.
  if (NumRegisteredSignals != 0) return;

  std::for_each(IntSigs, IntSigsEnd, RegisterHandler);
  std::for_each(KillSigs, KillSigsEnd, RegisterHandler);
}

static void UnregisterHandlers() {
  // Restore all of the signal handlers to how they were before we showed up.
  for (unsigned i = 0, e = NumRegisteredSignals; i != e; ++i)
    sigaction(RegisteredSignalInfo[i].SigNo,
              &RegisteredSignalInfo[i].SA, 0);
  NumRegisteredSignals = 0;
}


/// RemoveFilesToRemove - Process the FilesToRemove list. This function
/// should be called with the SignalsMutex lock held.
static void RemoveFilesToRemove() {
  while (!FilesToRemove.empty()) {
    FilesToRemove.back().eraseFromDisk(true);
    FilesToRemove.pop_back();
  }
}

// SignalHandler - The signal handler that runs.
static RETSIGTYPE SignalHandler(int Sig) {
  // Restore the signal behavior to default, so that the program actually
  // crashes when we return and the signal reissues.  This also ensures that if
  // we crash in our signal handler that the program will terminate immediately
  // instead of recursing in the signal handler.
  UnregisterHandlers();

  // Unmask all potentially blocked kill signals.
  sigset_t SigMask;
  sigfillset(&SigMask);
  sigprocmask(SIG_UNBLOCK, &SigMask, 0);

  SignalsMutex.acquire();
  RemoveFilesToRemove();

  if (std::find(IntSigs, IntSigsEnd, Sig) != IntSigsEnd) {
    if (InterruptFunction) {
      void (*IF)() = InterruptFunction;
      SignalsMutex.release();
      InterruptFunction = 0;
      IF();        // run the interrupt function.
      return;
    }
    
    SignalsMutex.release();
    raise(Sig);   // Execute the default handler.
    return;
  }

  SignalsMutex.release();

  // Otherwise if it is a fault (like SEGV) run any handler.
  for (unsigned i = 0, e = CallBacksToRun.size(); i != e; ++i)
    CallBacksToRun[i].first(CallBacksToRun[i].second);
}

void llvm::sys::RunInterruptHandlers() {
  SignalsMutex.acquire();
  RemoveFilesToRemove();
  SignalsMutex.release();
}

void llvm::sys::SetInterruptFunction(void (*IF)()) {
  SignalsMutex.acquire();
  InterruptFunction = IF;
  SignalsMutex.release();
  RegisterHandlers();
}

// RemoveFileOnSignal - The public API
bool llvm::sys::RemoveFileOnSignal(const sys::Path &Filename,
                                   std::string* ErrMsg) {
  SignalsMutex.acquire();
  FilesToRemove.push_back(Filename);

  SignalsMutex.release();

  RegisterHandlers();
  return false;
}

// DontRemoveFileOnSignal - The public API
void llvm::sys::DontRemoveFileOnSignal(const sys::Path &Filename) {
  SignalsMutex.acquire();
  std::vector<sys::Path>::reverse_iterator I =
    std::find(FilesToRemove.rbegin(), FilesToRemove.rend(), Filename);
  if (I != FilesToRemove.rend())
    FilesToRemove.erase(I.base()-1);
  SignalsMutex.release();
}

/// AddSignalHandler - Add a function to be called when a signal is delivered
/// to the process.  The handler can have a cookie passed to it to identify
/// what instance of the handler it is.
void llvm::sys::AddSignalHandler(void (*FnPtr)(void *), void *Cookie) {
  CallBacksToRun.push_back(std::make_pair(FnPtr, Cookie));
  RegisterHandlers();
}


// PrintStackTrace - In the case of a program crash or fault, print out a stack
// trace so that the user has an indication of why and where we died.
//
// On glibc systems we have the 'backtrace' function, which works nicely, but
// doesn't demangle symbols.  
static void PrintStackTrace(void *) {
#ifdef HAVE_BACKTRACE
  static void* StackTrace[256];
  // Use backtrace() to output a backtrace on Linux systems with glibc.
  int depth = backtrace(StackTrace,
                        static_cast<int>(array_lengthof(StackTrace)));
#if HAVE_DLFCN_H && __GNUG__
  int width = 0;
  for (int i = 0; i < depth; ++i) {
    Dl_info dlinfo;
    dladdr(StackTrace[i], &dlinfo);
    const char* name = strrchr(dlinfo.dli_fname, '/');

    int nwidth;
    if (name == NULL) nwidth = strlen(dlinfo.dli_fname);
    else              nwidth = strlen(name) - 1;

    if (nwidth > width) width = nwidth;
  }

  for (int i = 0; i < depth; ++i) {
    Dl_info dlinfo;
    dladdr(StackTrace[i], &dlinfo);

    fprintf(stderr, "%-2d", i);

    const char* name = strrchr(dlinfo.dli_fname, '/');
    if (name == NULL) fprintf(stderr, " %-*s", width, dlinfo.dli_fname);
    else              fprintf(stderr, " %-*s", width, name+1);

    fprintf(stderr, " %#0*lx",
            (int)(sizeof(void*) * 2) + 2, (unsigned long)StackTrace[i]);

    if (dlinfo.dli_sname != NULL) {
      int res;
      fputc(' ', stderr);
      char* d = abi::__cxa_demangle(dlinfo.dli_sname, NULL, NULL, &res);
      if (d == NULL) fputs(dlinfo.dli_sname, stderr);
      else           fputs(d, stderr);
      free(d);

      fprintf(stderr, " + %tu",(char*)StackTrace[i]-(char*)dlinfo.dli_saddr);
    }
    fputc('\n', stderr);
  }
#else
  backtrace_symbols_fd(StackTrace, depth, STDERR_FILENO);
#endif
#endif
}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIBABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void llvm::sys::PrintStackTraceOnErrorSignal() {
  AddSignalHandler(PrintStackTrace, 0);
}


/***/

// On Darwin, raise sends a signal to the main thread instead of the current
// thread. This has the unfortunate effect that assert() and abort() will end up
// bypassing our crash recovery attempts. We work around this for anything in
// the same linkage unit by just defining our own versions of the assert handler
// and abort.

#ifdef __APPLE__

void __assert_rtn(const char *func,
                  const char *file,
                  int line,
                  const char *expr) {
  if (func)
    fprintf(stderr, "Assertion failed: (%s), function %s, file %s, line %d.\n",
            expr, func, file, line);
  else
    fprintf(stderr, "Assertion failed: (%s), file %s, line %d.\n",
            expr, file, line);
  abort();
}

#include <signal.h>
#include <pthread.h>

void abort() {
  pthread_kill(pthread_self(), SIGABRT);
  usleep(1000);
  __builtin_trap();
}

#endif
