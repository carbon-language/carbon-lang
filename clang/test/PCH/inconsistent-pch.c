// Check that we don't crash in an inconsistent situation created by the stat
// cache.

// RUN: echo 'void f0(float *a0);' > %t.h
// RUN: clang-cc -emit-pch -o %t.h.pch %t.h
// RUN: rm %t.h
// RUN: not clang-cc -include-pch %t.h.pch %s 2> %t.err
// RUN: FileCheck %s < %t.err

// CHECK: inconsistent-pch.c:{{.*}}:{{.*}}: error: conflicting types for 'f0'
// CHECK: void f0(int *a0);
// CHECK: inconsistent-pch.c.tmp.h:{{.*}}:{{.*}}: note: previous declaration is here
// CHECK: 2 diagnostics generated.

void f0(int *a0);
