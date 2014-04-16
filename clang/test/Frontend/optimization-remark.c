// This file tests the -Rpass= flag with the inliner. The test is
// designed to always trigger the inliner, so it should be independent
// of the optimization level.

// RUN: %clang -c %s -Rpass=inline -O0 -gline-tables-only -S -o /dev/null 2> %t.err
// RUN: FileCheck < %t.err %s --check-prefix=INLINE

// RUN: %clang -c %s -Rpass=inline -O0 -S -o /dev/null 2> %t.err
// RUN: FileCheck < %t.err %s --check-prefix=INLINE-NO-LOC

int foo(int x, int y) __attribute__((always_inline));

int foo(int x, int y) { return x + y; }
int bar(int j) { return foo(j, j - 2); }

// INLINE: remark: foo inlined into bar [-Rpass]

// INLINE-NO-LOC: {{^remark: foo inlined into bar}}
// INLINE-NO-LOC: note: use -gline-tables-only -gcolumn-info to track
