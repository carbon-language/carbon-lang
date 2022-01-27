// RUN: c-index-test -test-load-source local %s -ivfsoverlay %t/does-not-exist.yaml > %t.stdout 2> %t.stderr
// RUN: FileCheck -check-prefix=STDERR %s < %t.stderr
// STDERR: fatal error: virtual filesystem overlay file '{{.*}}' not found
// RUN: FileCheck %s < %t.stdout
// CHECK: missing_vfs.c:[[@LINE+1]]:6: FunctionDecl=foo:[[@LINE+1]]:6
void foo(void);
