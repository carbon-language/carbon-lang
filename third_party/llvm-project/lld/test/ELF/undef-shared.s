# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s

# CHECK: error: undefined hidden symbol: hidden
# CHECK: >>> referenced by {{.*}}:(.data+0x0)
.global hidden
.hidden hidden

# CHECK: error: undefined internal symbol: internal
# CHECK: >>> referenced by {{.*}}:(.data+0x8)
.global internal
.internal internal

# CHECK: error: undefined protected symbol: protected
# CHECK: >>> referenced by {{.*}}:(.data+0x10)
.global protected
.protected protected

.section .data, "a"
 .quad hidden
 .quad internal
 .quad protected
