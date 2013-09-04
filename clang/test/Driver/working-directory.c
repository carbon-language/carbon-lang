// RUN: %clang -### -working-directory /no/such/dir/ input 2>&1 | FileCheck %s
// REQUIRES: shell-preserves-root

//CHECK: no such file or directory: '/no/such/dir/input'
