//===- FuzzerUtil.cpp - Misc utils ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Misc utils.
//===----------------------------------------------------------------------===//

#include "FuzzerInternal.h"
#include <iostream>
#include <sys/time.h>
#include <cassert>
#include <cstring>
#include <signal.h>

namespace fuzzer {

void Print(const Unit &v, const char *PrintAfter) {
  std::cerr << v.size() << ": ";
  for (auto x : v)
    std::cerr << (unsigned) x << " ";
  std::cerr << PrintAfter;
}

void PrintASCII(const Unit &U, const char *PrintAfter) {
  for (auto X : U)
    std::cerr << (char)((isascii(X) && X >= ' ') ? X : '?');
  std::cerr << PrintAfter;
}

std::string Hash(const Unit &in) {
  size_t h1 = 0, h2 = 0;
  for (auto x : in) {
    h1 += x;
    h1 *= 5;
    h2 += x;
    h2 *= 7;
  }
  return std::to_string(h1) + std::to_string(h2);
}

static void AlarmHandler(int, siginfo_t *, void *) {
  Fuzzer::AlarmCallback();
}

void SetTimer(int Seconds) {
  struct itimerval T {{Seconds, 0}, {Seconds, 0}};
  std::cerr << "SetTimer " << Seconds << "\n";
  int Res = setitimer(ITIMER_REAL, &T, nullptr);
  assert(Res == 0);
  struct sigaction sigact;
  memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = AlarmHandler;
  Res = sigaction(SIGALRM, &sigact, 0);
  assert(Res == 0);
}

}  // namespace fuzzer
