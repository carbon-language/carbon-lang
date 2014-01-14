// RUN: %clang_cc1 -triple i386-mingw32 -emit-llvm < %s | FileCheck %s

__attribute__((dllexport)) int bar1 = 2;
// CHECK-LABEL: @bar1 = dllexport global i32 2
__attribute__((dllimport)) extern int bar2;
// CHECK-LABEL: @bar2 = external dllimport global i32

void __attribute__((dllimport)) foo1();
void __attribute__((dllexport)) foo1(){}
// CHECK-LABEL: define dllexport void @foo1
void __attribute__((dllexport)) foo2();

// PR6269
__declspec(dllimport) void foo3();
__declspec(dllexport) void foo3(){}
// CHECK-LABEL: define dllexport void @foo3
__declspec(dllexport) void foo4();

__declspec(dllimport) void foo5();
// CHECK-LABEL: declare dllimport void @foo5

int use() { foo5(); return bar2; }
