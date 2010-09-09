// RUN: %clang_cc1 -fms-extensions -E %s | FileCheck %s

// Test that we properly expand the C99 _Pragma and Microsoft __pragma
// into #pragma directives, with newlines where needed. <rdar://problem/8412013>

// CHECK: extern
// CHECK: #line
// CHECK: #pragma warning(push)
// CHECK: #line
// CHECK: ; void f0();
// CHECK: #line
// CHECK: #pragma warning(pop)
// CHECK: #line
// CHECK: ; }
extern "C" { _Pragma("warning(push)"); void f0(); __pragma(warning(pop)); }
