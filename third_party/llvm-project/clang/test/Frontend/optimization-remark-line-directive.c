// This file tests -Rpass diagnostics together with #line
// directives. We cannot map #line directives back to
// a SourceLocation.

// The new PM inliner is not added to the default pipeline at O0, so we add
// some optimizations to trigger it.
// RUN: %clang_cc1 %s -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify

int foo(int x, int y) __attribute__((always_inline));
int foo(int x, int y) { return x + y; }

// expected-remark@+2 {{'foo' inlined into 'bar'}} expected-note@+2 {{could not determine the original source location for /bad/path/to/original.c:1230:25}}
#line 1230 "/bad/path/to/original.c"
int bar(int j) { return foo(j, j - 2); }
