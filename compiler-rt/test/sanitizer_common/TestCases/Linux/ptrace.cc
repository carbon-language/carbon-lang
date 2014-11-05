// RUN: %clangxx -O0 %s -o %t && %run %t

#include <assert.h>
#include <signal.h>
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
    int res;

#if __x86_64__
    user_regs_struct regs;
    res = ptrace(PTRACE_GETREGS, pid, NULL, &regs);
    assert(!res);
    if (regs.rip)
      printf("%zx\n", regs.rip);

    user_fpregs_struct fpregs;
    res = ptrace(PTRACE_GETFPREGS, pid, NULL, &fpregs);
    assert(!res);
    if (fpregs.mxcsr)
      printf("%x\n", fpregs.mxcsr);
#endif // __x86_64__

#if __powerpc64__
    struct pt_regs regs;
    res = ptrace((enum __ptrace_request)PTRACE_GETREGS, pid, NULL, &regs);
    assert(!res);
    if (regs.nip)
      printf("%lx\n", regs.nip);

    elf_fpregset_t fpregs;
    res = ptrace((enum __ptrace_request)PTRACE_GETFPREGS, pid, NULL, &fpregs);
    assert(!res);
    if ((elf_greg_t)fpregs[32]) // fpscr
      printf("%lx\n", (elf_greg_t)fpregs[32]);
#endif // __powerpc64__

    siginfo_t siginfo;
    res = ptrace(PTRACE_GETSIGINFO, pid, NULL, &siginfo);
    assert(!res);
    assert(siginfo.si_pid == pid);

    ptrace(PTRACE_CONT, pid, NULL, NULL);

    wait(NULL);
  }
  return 0;
}
