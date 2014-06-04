// RUN: %clangxx_asan -mllvm -asan-coverage=1 %s -o %t
// RUN: rm -rf %T/coverage-fork-direct
// RUN: mkdir -p %T/coverage-fork-direct && cd %T/coverage-fork-direct
// RUN: (ASAN_OPTIONS=coverage=1:coverage_direct=1:verbosity=1 %run %t; \
// RUN:  %sancov rawunpack *.sancov.raw; %sancov print *.sancov) 2>&1
//
// XFAIL: android

#include <stdio.h>
#include <string.h>
#include <unistd.h>

__attribute__((noinline))
void foo() { printf("foo\n"); }

__attribute__((noinline))
void bar() { printf("bar\n"); }

__attribute__((noinline))
void baz() { printf("baz\n"); }

int main(int argc, char **argv) {
  pid_t child_pid = fork();
  if (child_pid == 0) {
    fprintf(stderr, "Child PID: %d\n", getpid());
    baz();
  } else {
    fprintf(stderr, "Parent PID: %d\n", getpid());
    foo();
    bar();
  }
  return 0;
}

// CHECK-DAG: Child PID: [[ChildPID:[0-9]+]]
// CHECK-DAG: Parent PID: [[ParentPID:[0-9]+]]
// CHECK-DAG: read 3 PCs from {{.*}}.[[ParentPID]].sancov
// CHECK-DAG: read 1 PCs from {{.*}}.[[ChildPID]].sancov
