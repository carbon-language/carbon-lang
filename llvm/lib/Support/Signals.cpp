//===- Signals.cpp - Signal Handling support ------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// unix signals occuring while your program is running.
//
//===----------------------------------------------------------------------===//

#include "Support/Signals.h"
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <signal.h>

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


// SignalHandler - The signal handler that runs...
static void SignalHandler(int Sig) {
  while (!FilesToRemove.empty()) {
    std::remove(FilesToRemove.back().c_str());
    FilesToRemove.pop_back();
  }

  if (std::find(IntSigs, IntSigsEnd, Sig) != IntSigsEnd)
    exit(1);   // If this is an interrupt signal, exit the program

  // Otherwise if it is a fault (like SEGV) reissue the signal to die...
  signal(Sig, SIG_DFL);
}

static void RegisterHandler(int Signal) { signal(Signal, SignalHandler); }

// RemoveFileOnSignal - The public API
void RemoveFileOnSignal(const std::string &Filename) {
  FilesToRemove.push_back(Filename);

  std::for_each(IntSigs, IntSigsEnd, RegisterHandler);
  std::for_each(KillSigs, KillSigsEnd, RegisterHandler);
}
