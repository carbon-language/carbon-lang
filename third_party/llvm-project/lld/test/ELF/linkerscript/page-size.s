# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: ld.lld -z max-page-size=0x4000 %t -o %t2
# RUN: llvm-readobj -l %t2 | FileCheck %s

# CHECK:      ProgramHeaders [
# CHECK:        ProgramHeader {
# CHECK:          Type: PT_LOAD
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     VirtualAddress: 0x200000
# CHECK-NEXT:     PhysicalAddress: 0x200000
# CHECK-NEXT:     FileSize: 344
# CHECK-NEXT:     MemSize: 344
# CHECK-NEXT:     Flags [
# CHECK-NEXT:       PF_R
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 16384
# CHECK-NEXT:   }
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD
# CHECK-NEXT:     Offset: 0x158
# CHECK-NEXT:     VirtualAddress: 0x204158
# CHECK-NEXT:     PhysicalAddress: 0x204158
# CHECK-NEXT:     FileSize: 1
# CHECK-NEXT:     MemSize: 1
# CHECK-NEXT:     Flags [
# CHECK-NEXT:       PF_R
# CHECK-NEXT:       PF_X
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 16384
# CHECK-NEXT:   }
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD
# CHECK-NEXT:     Offset: 0x159
# CHECK-NEXT:     VirtualAddress: 0x208159
# CHECK-NEXT:     PhysicalAddress: 0x208159
# CHECK-NEXT:     FileSize: 8
# CHECK-NEXT:     MemSize: 8
# CHECK-NEXT:     Flags [
# CHECK-NEXT:       PF_R
# CHECK-NEXT:       PF_W
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 16384
# CHECK-NEXT:   }

# RUN: echo "SECTIONS { symbol = CONSTANT(MAXPAGESIZE); }" > %t.script
# RUN: ld.lld -z max-page-size=0x4000 -o %t1 --script %t.script %t
# RUN: llvm-readelf -s %t1 | FileCheck -check-prefix CHECK-SCRIPT %s

# CHECK-SCRIPT: 0000000000004000 0 NOTYPE GLOBAL DEFAULT ABS symbol

# RUN: not ld.lld -z max-page-size=0x1001 -o /dev/null --script %t.script %t 2>&1 \
# RUN:  | FileCheck -check-prefix=ERR1 %s
# ERR1: max-page-size: value isn't a power of 2

# RUN: not ld.lld -z max-page-size=-0x1000 -o /dev/null --script %t.script %t 2>&1 \
# RUN:  | FileCheck -check-prefix=ERR2 %s
# ERR2: invalid max-page-size: -0x1000

.global _start
_start:
  nop

.section .a, "aw"
.quad 0
