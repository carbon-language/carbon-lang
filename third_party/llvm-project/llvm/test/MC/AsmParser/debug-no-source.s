// UNSUPPORTED: -zos
// REQUIRES: default_triple
// RUN: llvm-mc %s | FileCheck %s

.file 1 "dir1/foo"

# CHECK-NOT: .file {{.*}} source
