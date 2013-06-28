// RUN: %clangxx_msan -m64 -O0 %s -o %t && %t

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
    res = ptrace(PTRACE_GETREGS, pid, NULL, &regs);
    assert(!res);
    if (regs.rip)
      printf("%zx\n", regs.rip);

    user_fpregs_struct fpregs;
    res = ptrace(PTRACE_GETFPREGS, pid, NULL, &fpregs);
    assert(!res);
    if (fpregs.mxcsr)
      printf("%x\n", fpregs.mxcsr);

    ptrace(PTRACE_CONT, pid, NULL, NULL);
    wait(NULL);
  }
  return 0;
}
