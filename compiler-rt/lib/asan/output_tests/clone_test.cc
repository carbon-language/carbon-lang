#ifdef __linux__
#include <stdio.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int Child(void *arg) {
  char x[32] = {0};  // Stack gets poisoned.
  printf("Child:  %p\n", x);
  _exit(1);  // NoReturn, stack will remain unpoisoned unless we do something.
}

int main(int argc, char **argv) {
  const int kStackSize = 1 << 20;
  char child_stack[kStackSize + 1];
  char *sp = child_stack + kStackSize;  // Stack grows down.
  printf("Parent: %p\n", sp);
  pid_t clone_pid = clone(Child, sp, CLONE_FILES | CLONE_VM, NULL, 0, 0, 0);
  waitpid(clone_pid, NULL, 0);
  for (int i = 0; i < kStackSize; i++)
    child_stack[i] = i;
  int ret = child_stack[argc - 1];
  printf("PASSED\n");
  return ret;
}
#else  // not __linux__
#include <stdio.h>
int main() {
  printf("PASSED\n");
  // Check-Common: PASSED
}
#endif
