// REQUIRES: x86-registered-target

// RUN: %clang -target x86_64-unknown-linux-gnu -x c -o %t1.so -emit-interface-stubs %s %S/object.c %S/weak.cpp && \
// RUN: llvm-nm %t1.so 2>&1 | FileCheck --check-prefix=CHECK-IFS %s

// RUN: %clang -target x86_64-unknown-linux-gnu -x c -o %t2.so -shared %s %S/object.c %S/weak.cpp && \
// RUN: llvm-nm %t2.so 2>&1 | FileCheck --check-prefix=CHECK-SO %s

// CHECK-IFS-DAG: data
// CHECK-IFS-DAG: foo
// CHECK-IFS-DAG: strongFunc
// CHECK-IFS-DAG: weakFunc

// CHECK-SO-DAG: data
// CHECK-SO-DAG: foo
// CHECK-SO-DAG: strongFunc
// CHECK-SO-DAG: weakFunc

int foo(int bar) { return 42 + 1844; }
