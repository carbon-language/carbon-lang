// RUN: %clangxx -fsanitize=returns-nonnull-attribute %s -O3 -o %t
// RUN: %run %t foo
// RUN: %run %t 2>&1 | FileCheck %s

__attribute__((returns_nonnull)) char *foo(char *a);

char *foo(char *a) {
  return a;
  // CHECK: nonnull.cpp:[[@LINE+2]]:1: runtime error: null pointer returned from function declared to never return null
  // CHECK-NEXT: nonnull.cpp:[[@LINE-5]]:16: note: returns_nonnull attribute specified here
}

int main(int argc, char **argv) {
  return foo(argv[1]) == 0;
}
