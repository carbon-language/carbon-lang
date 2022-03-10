// REQUIRES: arm
// RUN: llvm-mc  -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// The output file is large, most of it zeroes. We dissassemble only the
// parts we need to speed up the test and avoid a large output file
// RUN: llvm-objdump -d %t2 --start-address=0x100000 --stop-address=0x10000c | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump -d %t2 --start-address=0x200000 --stop-address=0x200002 | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump -d %t2 --start-address=0x300000 --stop-address=0x300002 | FileCheck --check-prefix=CHECK3 %s
// RUN: llvm-objdump -d %t2 --start-address=0x400000 --stop-address=0x400006 | FileCheck --check-prefix=CHECK4 %s
// RUN: llvm-objdump -d %t2 --start-address=0x1000000 --stop-address=0x1000036 | FileCheck --check-prefix=CHECK5 %s
// RUN: llvm-objdump -d %t2 --start-address=0x1100000 --stop-address=0x1100010 | FileCheck --check-prefix=CHECK6 %s
// RUN: llvm-objdump -d %t2 --start-address=0x1400004 --stop-address=0x140000c | FileCheck --check-prefix=CHECK7 %s
// RUN: llvm-objdump -d %t2 --start-address=0x1e00000 --stop-address=0x1e00006 | FileCheck --check-prefix=CHECK8 %s
// RUN: llvm-objdump -d %t2 --start-address=0x2200000 --stop-address=0x220000a | FileCheck --check-prefix=CHECK9 %s
// RUN: llvm-objdump -d %t2 --start-address=0x2300000 --stop-address=0x230000a | FileCheck --check-prefix=CHECK10 %s

// Test the Range extension Thunks for Thumb when all the code is in a single
// OutputSection. The Thumb unconditional branch b.w and branch and link bl
// instructions have a range of 16Mb. We create a series of Functions a
// megabyte apart. We expect range extension thunks to be created when a
// branch is out of range. Thunks will be reused whenever they are in range
 .syntax unified

// Define a function aligned on a megabyte boundary
 .macro FUNCTION suff
 .section .text.\suff\(), "ax", %progbits
 .thumb
 .balign 0x100000
 .globl tfunc\suff\()
 .type  tfunc\suff\(), %function
tfunc\suff\():
 bx lr
 .endm

 .section .text, "ax", %progbits
 .thumb
 .globl _start
_start:
// tfunc00 and tfunc15 are within 16Mb no Range Thunks expected
 bl tfunc00
 bl tfunc15
// tfunc16 is > 16Mb away, expect a Range Thunk to be generated, to go into
// the first of the pre-created ThunkSections.
 bl tfunc16
// CHECK1: Disassembly of section .text:
// CHECK1-EMPTY:
// CHECK1-NEXT: <_start>:
// CHECK1-NEXT:   100000:       ff f0 fe ff     bl      0x200000 <tfunc00>
// CHECK1-NEXT:   100004:       ff f3 fc d7     bl      0x1100000 <tfunc15>
// CHECK1-NEXT:   100008:       ff f2 fc d7     bl      0x1000004 <__Thumbv7ABSLongThunk_tfunc16>

 FUNCTION 00
// CHECK2:  <tfunc00>:
// CHECK2-NEXT:   200000:       70 47   bx      lr
        FUNCTION 01
// CHECK3: <tfunc01>:
// CHECK3-NEXT:   300000:       70 47   bx      lr
 FUNCTION 02
// tfunc28 is > 16Mb away, expect a Range Thunk to be generated, to go into
// the first of the pre-created ThunkSections.
        b.w tfunc28
// CHECK4: <tfunc02>:
// CHECK4-NEXT:   400000:       70 47   bx      lr
// CHECK4-NEXT:   400002:       00 f0 01 90     b.w     0x1000008 <__Thumbv7ABSLongThunk_tfunc28>
 FUNCTION 03
 FUNCTION 04
 FUNCTION 05
 FUNCTION 06
 FUNCTION 07
 FUNCTION 08
 FUNCTION 09
 FUNCTION 10
 FUNCTION 11
 FUNCTION 12
 FUNCTION 13
 FUNCTION 14
// Expect precreated ThunkSection here
// CHECK5: <__Thumbv7ABSLongThunk_tfunc16>:
// CHECK5-NEXT:  1000004:       ff f1 fc bf     b.w     0x1200000 <tfunc16>
// CHECK5: <__Thumbv7ABSLongThunk_tfunc28>:
// CHECK5-NEXT:  1000008:       ff f1 fa 97     b.w     0x1e00000 <tfunc28>
// CHECK5: <__Thumbv7ABSLongThunk_tfunc32>:
// CHECK5-NEXT:  100000c:       40 f2 01 0c     movw    r12, #1
// CHECK5-NEXT:  1000010:       c0 f2 20 2c     movt    r12, #544
// CHECK5-NEXT:  1000014:       60 47   bx      r12
// CHECK5: <__Thumbv7ABSLongThunk_tfunc33>:
// CHECK5-NEXT:  1000016:       40 f2 01 0c     movw    r12, #1
// CHECK5-NEXT:  100001a:       c0 f2 30 2c     movt    r12, #560
// CHECK5-NEXT:  100001e:       60 47   bx      r12
// CHECK5: <__Thumbv7ABSLongThunk_tfunc02>:
// CHECK5-NEXT:  1000020:       ff f7 ee 97     b.w     0x400000 <tfunc02>
 FUNCTION 15
// tfunc00 and tfunc01 are < 16Mb away, expect no range extension thunks
 bl tfunc00
 bl tfunc01
// tfunc32 and tfunc33 are > 16Mb away, expect range extension thunks in the
// precreated thunk section
 bl tfunc32
 bl tfunc33
// CHECK6:  <tfunc15>:
// CHECK6-NEXT:  1100000:       70 47   bx      lr
// CHECK6-NEXT:  1100002:       ff f4 fd d7     bl      0x200000 <tfunc00>
// CHECK6-NEXT:  1100006:       ff f5 fb d7     bl      0x300000 <tfunc01>
// CHECK6-NEXT:  110000a:       ff f6 ff ff     bl      0x100000c <__Thumbv7ABSLongThunk_tfunc32>
// CHECK6-NEXT:  110000e:       00 f7 02 f8     bl      0x1000016 <__Thumbv7ABSLongThunk_tfunc33>
 FUNCTION 16
 FUNCTION 17
 FUNCTION 18
// Expect another precreated thunk section here
// CHECK7: <__Thumbv7ABSLongThunk_tfunc15>:
// CHECK7-NEXT:  1400004:       ff f4 fc bf     b.w     0x1100000 <tfunc15>
// CHECK7: <__Thumbv7ABSLongThunk_tfunc16>:
// CHECK7-NEXT:  1400008:       ff f5 fa bf     b.w     0x1200000 <tfunc16>
 FUNCTION 19
 FUNCTION 20
 FUNCTION 21
 FUNCTION 22
 FUNCTION 23
 FUNCTION 24
 FUNCTION 25
 FUNCTION 26
 FUNCTION 27
 FUNCTION 28
// tfunc02 is > 16Mb away, expect range extension thunks in precreated thunk
// section
// CHECK8:  <tfunc28>:
// CHECK8-NEXT:  1e00000:       70 47   bx      lr
// CHECK8-NEXT:  1e00002:       00 f6 0d 90     b.w     0x1000020 <__Thumbv7ABSLongThunk_tfunc02>

 b.w tfunc02
 FUNCTION 29
 FUNCTION 30
 FUNCTION 31
 FUNCTION 32
 // tfunc15 and tfunc16 are > 16 Mb away expect Thunks in the nearest
 // precreated thunk section.
 bl tfunc15
 bl tfunc16
// CHECK9: <tfunc32>:
// CHECK9:  2200000:    70 47   bx      lr
// CHECK9-NEXT:  2200002:       ff f5 ff d7     bl      0x1400004 <__Thumbv7ABSLongThunk_tfunc15>
// CHECK9-NEXT:  2200006:       ff f5 ff d7     bl      0x1400008 <__Thumbv7ABSLongThunk_tfunc16>

 FUNCTION 33
 bl tfunc15
 bl tfunc16
// CHECK10: <tfunc33>:
// CHECK10:  2300000:   70 47   bx      lr
// CHECK10-NEXT:  2300002:      ff f4 ff d7     bl      0x1400004 <__Thumbv7ABSLongThunk_tfunc15>
// CHECK10-NEXT:  2300006:      ff f4 ff d7     bl      0x1400008 <__Thumbv7ABSLongThunk_tfunc16>
