// RUN: %clang_cc1 -fms-extensions -fblocks -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s

void f1(const char* a, const char* b) {}
// CHECK: "\01?f1@@YAXPBD0@Z"

void f2(const char* a, char* b) {}
// CHECK: "\01?f2@@YAXPBDPAD@Z"

void f3(int a, const char* b, const char* c) {}
// CHECK: "\01?f3@@YAXHPBD0@Z"

const char *f4(const char* a, const char* b) {}
// CHECK: "\01?f4@@YAPBDPBD0@Z"

// FIXME: tests for more than 10 types?

struct S {};

void g4(const char* a, struct S* b, const char *c, struct S* d) {}
// CHECK: "\01?g4@@YAXPBDPAUS@@01@Z"

typedef void (*VoidFunc)();

void foo_ptr(const char* a, const char* b, VoidFunc c, VoidFunc d) {}
// CHECK: @"\01?foo_ptr@@YAXPBD0P6AXXZ1@Z"

// Make sure that different aliases of built-in types end up mangled as the
// built-ins.
typedef unsigned int uintptr_t;
typedef unsigned int size_t;
void *h(size_t a, uintptr_t b) {}
// CHECK: "\01?h@@YAPAXII@Z"
