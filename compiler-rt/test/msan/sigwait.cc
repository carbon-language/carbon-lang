// RUN: %clangxx_msan -std=c++11 -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <sanitizer/msan_interface.h>
#include <signal.h>
#include <sys/time.h>
#include <unistd.h>

void test_sigwait() {
  sigset_t s;
  sigemptyset(&s);
  sigaddset(&s, SIGUSR1);
  sigprocmask(SIG_BLOCK, &s, 0);

  if (pid_t pid = fork()) {
    kill(pid, SIGUSR1);
    _exit(0);
  } else {
    int sig;
    int res = sigwait(&s, &sig);
    assert(!res);
    // The following checks that sig is initialized.
    assert(sig == SIGUSR1);
  }
}

int main(void) {
  test_sigwait();
  return 0;
}
