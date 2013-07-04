// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -x c++ %s 2>&1 | FileCheck %s

// expected-note@+1 5{{previous definition is here}}
int main() {
  return 0;
}

// expected-error@+3 {{conflicting types for 'main}}
// expected-warning@+2 {{return type of 'main' is not 'int'}}
// expected-note@+1 {{change return type to 'int'}}
void main() {
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:1-[[@LINE-1]]:5}:"int"
}

// expected-error@+3 {{conflicting types for 'main}}
// expected-warning@+2 {{return type of 'main' is not 'int'}}
// expected-note@+1 {{change return type to 'int'}}
double main() {
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:1-[[@LINE-1]]:7}:"int"
  return 0.0;
}

// Currently we suggest to replace only 'float' here because we don't store
// enough source locations.
//
// expected-error@+3 {{conflicting types for 'main}}
// expected-warning@+2 {{return type of 'main' is not 'int'}}
// expected-note@+1 {{change return type to 'int'}}
const float main() {
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:12}:"int"
  return 0.0f;
}

typedef void *(*fptr)(int a);

// expected-error@+2 {{conflicting types for 'main}}
// expected-warning@+1 {{return type of 'main' is not 'int'}}
fptr main() {
  return (fptr) 0;
}

// expected-error@+2 {{conflicting types for 'main}}
// expected-warning@+1 {{return type of 'main' is not 'int'}}
void *(*main())(int a) {
  return (fptr) 0;
}

