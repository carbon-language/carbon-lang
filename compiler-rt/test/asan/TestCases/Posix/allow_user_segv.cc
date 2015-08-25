// Regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=180

// RUN: %clangxx_asan -O0 %s -o %t && %env_asan_opts=allow_user_segv_handler=true not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && %env_asan_opts=allow_user_segv_handler=true not %run %t 2>&1 | FileCheck %s

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

struct sigaction original_sigaction_sigbus;
struct sigaction original_sigaction_sigsegv;

void User_OnSIGSEGV(int signum, siginfo_t *siginfo, void *context) {
  fprintf(stderr, "User sigaction called\n");
  struct sigaction original_sigaction;
  if (signum == SIGBUS)
    original_sigaction = original_sigaction_sigbus;
  else if (signum == SIGSEGV)
    original_sigaction = original_sigaction_sigsegv;
  else {
    printf("Invalid signum");
    exit(1);
  }
  if (original_sigaction.sa_flags | SA_SIGINFO)
    original_sigaction.sa_sigaction(signum, siginfo, context);
  else
    original_sigaction.sa_handler(signum);
}

int DoSEGV() {
  volatile int *x = 0;
  return *x;
}

int InstallHandler(int signum, struct sigaction *original_sigaction) {
  struct sigaction user_sigaction;
  user_sigaction.sa_sigaction = User_OnSIGSEGV;
  user_sigaction.sa_flags = SA_SIGINFO;
  if (sigaction(signum, &user_sigaction, original_sigaction)) {
    perror("sigaction");
    return 1;
  }
  return 0;
}

int main() {
  // Let's install handlers for both SIGSEGV and SIGBUS, since pre-Yosemite
  // 32-bit Darwin triggers SIGBUS instead.
  if (InstallHandler(SIGSEGV, &original_sigaction_sigsegv)) return 1;
  if (InstallHandler(SIGBUS, &original_sigaction_sigbus)) return 1;
  fprintf(stderr, "User sigaction installed\n");
  return DoSEGV();
}

// CHECK: User sigaction installed
// CHECK-NEXT: User sigaction called
// CHECK-NEXT: ASAN:DEADLYSIGNAL
// CHECK: AddressSanitizer: SEGV on unknown address
