// RUN: %clangxx -O1 %s -o %t && %env_tool_opts=handle_sigtrap=1 %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <signal.h>
#include <stdio.h>

int handled;

void handler(int signo, siginfo_t *info, void *uctx) {
  handled = 1;
}

int main() {
  struct sigaction a = {}, old = {};
  a.sa_sigaction = handler;
  a.sa_flags = SA_SIGINFO;
  sigaction(SIGTRAP, &a, &old);

  a = {};
  sigaction(SIGTRAP, 0, &a);
  assert(a.sa_sigaction == handler);
  assert(a.sa_flags & SA_SIGINFO);

  __builtin_debugtrap();
  assert(handled);
  fprintf(stderr, "HANDLED %d\n", handled);
}

// CHECK: HANDLED 1
