// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++1y -o - %s 2>&1 | FileCheck %s
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++1y -fdefine-sized-deallocation -o - %s 2>&1 | FileCheck %s
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++11 -fsized-deallocation -o - %s 2>&1 | FileCheck %s
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++11 -fsized-deallocation -fdefine-sized-deallocation -o - %s 2>&1 | FileCheck %s
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++11 -o - %s 2>&1 | FileCheck %s

// CHECK-UNSIZED-NOT: _ZdlPvm
// CHECK-UNSIZED-NOT: _ZdaPvm

void operator delete(void*, unsigned long) throw() __attribute__((alias("foo")));
extern "C" void foo(void*, unsigned long) {}

// CHECK-DAG: @_ZdlPvm = alias void (i8*, i64)* @foo
