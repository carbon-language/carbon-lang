// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t.o
// RUN: not ld.lld -shared %t.o -o /dev/null 2>&1 | FileCheck %s
// CHECK: R_AARCH64_PREL16 cannot be used against symbol 'foo'; recompile with -fPIC
// CHECK: >>> defined in {{.*}}
// CHECK: >>> referenced by {{.*}}:(.data+0x0)

.data
  .hword foo - .
