// RUN: %clang_cc1 -std=c++11 -ast-print -fms-extensions %s | FileCheck %s
// FIXME: align attribute print

// CHECK: int x __attribute__((aligned(4, 0)));
int x __attribute__((aligned(4)));

// FIXME: Print this at a valid location for a __declspec attr.
// CHECK: int y __declspec(align(4, 1));
__declspec(align(4)) int y;

// CHECK: gnu::aligned(4, 0)]];
int z [[gnu::aligned(4)]];

// CHECK: __attribute__((deprecated("warning")));
int a __attribute__((deprecated("warning")));

// CHECK: gnu::deprecated("warning")]];
int b [[gnu::deprecated("warning")]];

// CHECK: void foo() __attribute__((const));
void foo() __attribute__((const));

// CHECK: void bar() __attribute__((__const));
void bar() __attribute__((__const));

// CHECK: int f1() __attribute__((warn_unused_result));
int f1() __attribute__((warn_unused_result));

// CHECK: clang::warn_unused_result]];
int f2 [[clang::warn_unused_result]] ();

// CHECK: gnu::warn_unused_result]];
int f3 [[gnu::warn_unused_result]] ();

// FIXME: ast-print need to print C++11
// attribute after function declare-id.
// CHECK: noreturn]];
void f4 [[noreturn]] ();

// CHECK: std::noreturn]];
void f5 [[std::noreturn]] ();

// CHECK: __attribute__((gnu_inline));
inline void f6() __attribute__((gnu_inline));

// CHECK: gnu::gnu_inline]];
inline void f7 [[gnu::gnu_inline]] ();

// arguments printing
// CHECK: __attribute__((format("printf", 2, 3)));
void f8 (void *, const char *, ...) __attribute__ ((format (printf, 2, 3)));
