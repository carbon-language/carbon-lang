// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++1y -o - %s 2>&1 | FileCheck %s  --check-prefix=CHECKUND
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++1y -fdefine-sized-deallocation -o - %s 2>&1 | FileCheck %s --check-prefix=CHECKDEF
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++1y -fno-sized-deallocation -o - %s 2>&1 | FileCheck %s --check-prefix=CHECKNO
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++11 -fsized-deallocation -o - %s 2>&1 | FileCheck %s --check-prefix=CHECKUND
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++11 -fsized-deallocation -fdefine-sized-deallocation -o - %s 2>&1 | FileCheck %s --check-prefix=CHECKDEF
// RUN: %clang -cc1 -emit-llvm -triple x86_64-unknown-unknown -std=c++11 -o - %s 2>&1 | FileCheck %s --check-prefix=CHECKNO

void operator delete(void*, unsigned long) throw() __attribute__((alias("foo")));
extern "C" void foo(void*, unsigned long) {}

// CHECKUND-DAG: @_ZdlPvm = weak alias void (i8*, i64)* @foo
// CHECKDEF-DAG: @_ZdlPvm = alias void (i8*, i64)* @foo
// CHECKNO-DAG: @_ZdlPvm = alias void (i8*, i64)* @foo
