// RUN: %clang -x c -o libfoo.so -emit-interface-stubs %s %S/object.c %S/weak.cpp && \
// RUN: llvm-nm libfoo.so 2>&1 | FileCheck %s

// RUN: %clang -x c -o libfoo.so -shared %s %S/object.c %S/weak.cpp && \
// RUN: llvm-nm libfoo.so 2>&1 | FileCheck %s

// CHECK-DAG: data
// CHECK-DAG: foo
// CHECK-DAG: strongFunc
// CHECK-DAG: weakFunc

int foo(int bar) { return 42 + 1844; }