// Regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=180

// RUN: %clangxx_asan -m64 -O0 %s -o %t && ASAN_OPTIONS=allow_user_segv_handler=true %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m64 -O2 %s -o %t && ASAN_OPTIONS=allow_user_segv_handler=true %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m32 -O0 %s -o %t && ASAN_OPTIONS=allow_user_segv_handler=true %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m32 -O2 %s -o %t && ASAN_OPTIONS=allow_user_segv_handler=true %t 2>&1 | FileCheck %s

#include <signal.h>
#include <stdio.h>

struct sigaction user_sigaction;
struct sigaction original_sigaction;

void User_OnSIGSEGV(int signum, siginfo_t *siginfo, void *context) {
  fprintf(stderr, "User sigaction called\n");
  if (original_sigaction.sa_flags | SA_SIGINFO)
    original_sigaction.sa_sigaction(signum, siginfo, context);
  else
    original_sigaction.sa_handler(signum);
}

int DoSEGV() {
  volatile int *x = 0;
  return *x;
}

int main() {
  user_sigaction.sa_sigaction = User_OnSIGSEGV;
  user_sigaction.sa_flags = SA_SIGINFO;
  if (sigaction(SIGSEGV, &user_sigaction, &original_sigaction)) {
    perror("sigaction");
    return 1;
  }
  fprintf(stderr, "User sigaction installed\n");
  return DoSEGV();
}

// CHECK: User sigaction installed
// CHECK-NEXT: User sigaction called
// CHECK-NEXT: ASAN:SIGSEGV
// CHECK: AddressSanitizer: SEGV on unknown address
