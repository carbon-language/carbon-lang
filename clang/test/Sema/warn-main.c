// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -x c++ %s 2>&1 | FileCheck %s

// expected-note@+1 2{{previous definition is here}}
int main() {
  return 0;
}

// expected-error@+2 {{static declaration of 'main' follows non-static declaration}}
// expected-warning@+1 {{'main' should not be declared static}}
static int main() {
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:1-[[@LINE-1]]:8}:""
  return 0;
}

// expected-error@+3 {{redefinition of 'main'}}
// expected-error@+2 {{'main' is not allowed to be declared inline}}
// expected-note@+1 {{previous definition is here}}
inline int main() {
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:1-[[@LINE-1]]:8}:""
  return 0;
}

// expected-warning@+6 {{function 'main' declared 'noreturn' should not return}}
// expected-error@+3 {{redefinition of 'main'}}
// expected-warning@+2 {{'main' is not allowed to be declared _Noreturn}}
// expected-note@+1 {{remove '_Noreturn'}}
_Noreturn int main() {
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:1-[[@LINE-1]]:11}:""
  return 0;
}

