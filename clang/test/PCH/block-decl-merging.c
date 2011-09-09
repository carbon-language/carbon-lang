// RUN: %clang_cc1 -triple i386-apple-darwin10 -fblocks %s -emit-pch -o %t
// RUN: %clang_cc1 -triple i386-apple-darwin10 -fblocks %s -include-pch %t -emit-llvm -o - | \
// RUN:   FileCheck %s

#ifndef HEADER
#define HEADER

// CHECK: @_NSConcreteGlobalBlock = extern_weak global
extern void * _NSConcreteStackBlock[32] __attribute__((weak_import));
// CHECK: @_NSConcreteStackBlock = extern_weak global
extern void * _NSConcreteGlobalBlock[32] __attribute__((weak_import));
extern void _Block_object_dispose(const void *, const int) __attribute__((weak_import));
// CHECK: declare extern_weak void @_Block_object_assign
extern void _Block_object_assign(void *, const void *, const int) __attribute__((weak_import));
// CHECK: declare extern_weak void @_Block_object_dispose

#else

void *x = ^(){};

void f1(void (^a0)(void));

void f0() {
  __block int x;
  f1(^(void){ x = 1; });
}

#endif
