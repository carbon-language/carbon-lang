// RUN: echo 'typedef int t0;' > %t.pfx.h
// RUN: clang -x objective-c-header %t.pfx.h -o %t.pfx.h.gch
// RUN: c-index-test -test-load-source local %s -include %t.pfx.h > %t
// RUN: FileCheck %s < %t
// CHECK: cindex-from-source.m:{{.*}}:{{.*}}: StructDecl=s0:{{.*}}:{{.*}} [Context=cindex-from-source.m]
// CHECK: cindex-from-source.m:{{.*}}:{{.*}}: VarDecl=g0:{{.*}}:{{.*}} [Context=cindex-from-source.m]

struct s0 {};
t0 g0;
