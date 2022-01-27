// RUN: %check_clang_tidy %s bugprone-signal-handler %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: bugprone-signal-handler.AsyncSafeFunctionSet, value: "POSIX"}]}' \
// RUN: -- -isystem %S/Inputs/Headers

#include "signal.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "unistd.h"

void handler_bad(int) {
  printf("1234");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
}

void handler_good(int) {
  abort();
  _Exit(0);
  _exit(0);
  quick_exit(0);
  signal(0, SIG_DFL);
  memcpy((void*)10, (const void*)20, 1);
}

void test() {
  signal(SIGINT, handler_good);
  signal(SIGINT, handler_bad);
}
