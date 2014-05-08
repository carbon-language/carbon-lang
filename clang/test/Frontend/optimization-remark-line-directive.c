// This file tests -Rpass diagnostics together with #line
// directives. We cannot map #line directives back to
// a SourceLocation.

// RUN: %clang -c %s -Rpass=inline -O0 -S -gmlt -o /dev/null 2> %t.err
// RUN: FileCheck < %t.err %s --check-prefix=INLINE-INVALID-LOC
//
int foo(int x, int y) __attribute__((always_inline));
int foo(int x, int y) { return x + y; }

#line 1230 "/bad/path/to/original.c"
int bar(int j) { return foo(j, j - 2); }

// INLINE-INVALID-LOC: {{^remark: foo inlined into bar}}
// INLINE-INVALID-LOC: note: could not determine the original source location for /bad/path/to/original.c:1230:0
