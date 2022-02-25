// REQUIRES: powerpc-registered-target
// RUN: not %clang -target powerpc64-unknown-freebsd -maix-struct-return \
// RUN:   -c %s 2>&1 | FileCheck %s
// RUN: not %clang -target powerpc64-unknown-freebsd -msvr4-struct-return \
// RUN:   -c %s 2>&1 | FileCheck %s
// RUN: not %clang -target powerpc64le-unknown-linux -maix-struct-return \
// RUN:   -c %s 2>&1 | FileCheck %s
// RUN: not %clang -target powerpc64le-unknown-linux -msvr4-struct-return \
// RUN:   -c %s 2>&1 | FileCheck %s
// CHECK: unsupported option
