# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld2 %t -o %t2
# RUN: llvm-readobj -sections -program-headers %t2 | FileCheck %s
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

# CHECK:        Name: .r
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Offset: 0x1000
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }

# CHECK:      ProgramHeaders [
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     VirtualAddress:
# CHECK-NEXT:     PhysicalAddress:
# CHECK-NEXT:     FileSize: 4104
# CHECK-NEXT:     MemSize: 4104
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
