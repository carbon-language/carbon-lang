// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld --gc-sections %t.o -o %t
// RUN: llvm-readobj -s -section-data %t | FileCheck %s

// Here we check that if all FDEs referencing a CIE
// were removed, CIE is also removed.
// CHECK:        Section {
// CHECK:        Index:
// CHECK:        Name: .eh_frame
// CHECK-NEXT:   Type: SHT_X86_64_UNWIND
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_ALLOC
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: 0x10120
// CHECK-NEXT:   Offset: 0x120
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Link: 0
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: 8
// CHECK-NEXT:   EntrySize: 0
// CHECK-NEXT:   SectionData (
// CHECK-NEXT:   )
// CHECK-NEXT: }

.section foo,"ax",@progbits
.cfi_startproc
 nop
.cfi_endproc

.section bar,"ax",@progbits
.cfi_startproc
 nop
 nop
.cfi_endproc

.text
.globl _start;
_start:
