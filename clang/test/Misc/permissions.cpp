// REQUIRES: shell

// MSYS doesn't emulate umask.
// FIXME: Could we introduce another feature for it?
// REQUIRES: shell-preserves-root'

// RUN: umask 000
// RUN: %clang_cc1 -emit-llvm-bc %s -o %t
// RUN: ls -l %t | FileCheck --check-prefix=CHECK000 %s
// CHECK000: rw-rw-rw-

// RUN: umask 002
// RUN: %clang_cc1 -emit-llvm-bc %s -o %t
// RUN: ls -l %t | FileCheck --check-prefix=CHECK002 %s
// CHECK002: rw-rw-r--
