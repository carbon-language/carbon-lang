// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
//
// RUN: %clangxx_asan -O0 %s -o %t && %run %t
// RUN: %clangxx_asan -DPOSITIVE -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// REQUIRES: x86_64-supported-target,i386-supported-target

#include <assert.h>
#include <stdio.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
  pid_t pid;
  pid = fork();
  if (pid == 0) { // child
    ptrace(PTRACE_TRACEME, 0, NULL, NULL);
    execl("/bin/true", "true", NULL);
  } else {
    wait(NULL);
    user_regs_struct regs;
    int res;
    user_regs_struct * volatile pregs = &regs;
#ifdef POSITIVE
    ++pregs;
#endif
    res = ptrace(PTRACE_GETREGS, pid, NULL, pregs);
    // CHECK: AddressSanitizer: stack-buffer-overflow
    // CHECK: {{.*ptrace.cc:}}[[@LINE-2]]
    assert(!res);
#ifdef __x86_64__
    printf("%lx\n", (unsigned long)regs.rip);
#else
    printf("%lx\n", regs.eip);
#endif

    user_fpregs_struct fpregs;
    res = ptrace(PTRACE_GETFPREGS, pid, NULL, &fpregs);
    assert(!res);
    printf("%lx\n", (unsigned long)fpregs.cwd);

#ifndef __x86_64__
    user_fpxregs_struct fpxregs;
    res = ptrace(PTRACE_GETFPXREGS, pid, NULL, &fpxregs);
    assert(!res);
    printf("%lx\n", (unsigned long)fpxregs.mxcsr);
#endif

    ptrace(PTRACE_CONT, pid, NULL, NULL);
    wait(NULL);
  }
  return 0;
}
