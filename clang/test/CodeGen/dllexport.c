// RUN: %clang_cc1 -triple i686-pc-win32 -std=c99 -O2 -disable-llvm-optzns -emit-llvm < %s | FileCheck %s

#define DLLEXPORT __declspec(dllexport)

inline void DLLEXPORT f() {}
extern void DLLEXPORT f();

// CHECK: define weak_odr dllexport void @f()
