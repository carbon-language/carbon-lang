// RUN: not %clang_cc1 %s -triple=i686-pc-win32 -fms-extensions -emit-llvm-only 2>&1 | FileCheck %s

// Reduced from WebKit.

// FIXME: Implement this pragma and test the codegen.  We probably want to
// completely skip @llvm.global_ctors and just create global function pointers
// to the initializer with the right section.

// CHECK: '#pragma init_seg' not implemented
#pragma init_seg(".unwantedstaticinits")
struct A {
  A();
  ~A();
  int a;
};
A a;
