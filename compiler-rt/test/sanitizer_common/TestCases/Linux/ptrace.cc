// RUN: %clangxx -O0 %s -o %t && %run %t

#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <sys/uio.h>
#include <unistd.h>
#include <elf.h>
#if __mips64
 #include <asm/ptrace.h>
 #include <sys/procfs.h>
#endif
#ifdef __aarch64__
// GLIBC 2.20+ sys/user does not include asm/ptrace.h
 #include <asm/ptrace.h>
#endif

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

#if (__powerpc64__ || __mips64)
    struct pt_regs regs;
    res = ptrace((enum __ptrace_request)PTRACE_GETREGS, pid, NULL, &regs);
    assert(!res);
#if (__powerpc64__)
    if (regs.nip)
      printf("%lx\n", regs.nip);
#else
    if (regs.cp0_epc)
    printf("%lx\n", regs.cp0_epc);
#endif
    elf_fpregset_t fpregs;
    res = ptrace((enum __ptrace_request)PTRACE_GETFPREGS, pid, NULL, &fpregs);
    assert(!res);
    if ((elf_greg_t)fpregs[32]) // fpscr
      printf("%lx\n", (elf_greg_t)fpregs[32]);
#endif // (__powerpc64__ || __mips64)

#if (__aarch64__)
    struct iovec regset_io;

    struct user_pt_regs regs;
    regset_io.iov_base = &regs;
    regset_io.iov_len = sizeof(regs);
    res = ptrace(PTRACE_GETREGSET, pid, (void*)NT_PRSTATUS, (void*)&regset_io);
    assert(!res);
    if (regs.pc)
      printf("%llx\n", regs.pc);

    struct user_fpsimd_state fpregs;
    regset_io.iov_base = &fpregs;
    regset_io.iov_len = sizeof(fpregs);
    res = ptrace(PTRACE_GETREGSET, pid, (void*)NT_FPREGSET, (void*)&regset_io);
    assert(!res);
    if (fpregs.fpsr)
      printf("%x\n", fpregs.fpsr);
#endif // (__aarch64__)

    siginfo_t siginfo;
    res = ptrace(PTRACE_GETSIGINFO, pid, NULL, &siginfo);
    assert(!res);
    assert(siginfo.si_pid == pid);

    ptrace(PTRACE_CONT, pid, NULL, NULL);

    wait(NULL);
  }
  return 0;
}
