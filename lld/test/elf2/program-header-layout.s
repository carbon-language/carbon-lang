# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: lld -flavor gnu2 %t -o %t2
# RUN: llvm-readobj -program-headers %t2 | FileCheck %s
# REQUIRES: x86

# Check that different output sections with the same flags are merged into a
# single Read/Write PT_LOAD.

.section .r,"a"
.globl _start
_start:
.quad 0

.section .a,"aw"
.quad 1

.section .b,"aw"
.quad 2

# CHECK:      ProgramHeaders [
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     VirtualAddress:
# CHECK-NEXT:     PhysicalAddress:
# CHECK-NEXT:     FileSize:
# CHECK-NEXT:     MemSize:
# CHECK-NEXT:     Flags [
# CHECK-NEXT:       PF_R
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment:
# CHECK-NEXT:   }
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD
# CHECK-NEXT:     Offset:
# CHECK-NEXT:     VirtualAddress:
# CHECK-NEXT:     PhysicalAddress:
# CHECK-NEXT:     FileSize: 16
# CHECK-NEXT:     MemSize: 16
# CHECK-NEXT:     Flags [
# CHECK-NEXT:       PF_R
# CHECK-NEXT:       PF_W
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment:
# CHECK-NEXT:   }
# CHECK-NEXT: ]
