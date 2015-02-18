// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++1y %s -o -
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++1y %s -fdefine-sized-deallocation -o -
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++11 -fsized-deallocation %s -o -
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++11 -fsized-deallocation -fdefine-sized-deallocation %s -o -
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++11 %s -o -

// CHECK-UNSIZED-NOT: _ZdlPvm
// CHECK-UNSIZED-NOT: _ZdaPvm

void operator delete(void*, unsigned long) throw() __attribute__((alias("foo")));
extern "C" void foo(void*, unsigned long) {}

// CHECK-DAG: @_ZdlPvm = alias void (i8*, i64)* @my_delete
