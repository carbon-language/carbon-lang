// RUN: %clang_cc1 -triple s390x-unknown-linux -fms-extensions -fsyntax-only -verify %s

// expected-warning@+1{{unknown attribute 'ms_hook_prologue' ignored}}
int __attribute__((ms_hook_prologue)) foo(int a, int b) {
  return a+b;
}
