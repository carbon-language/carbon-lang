// REQUIRES: arm
// RUN: llvm-mc -filetype=obj --arm-add-build-attributes -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t --no-merge-exidx-entries -o %t2
// RUN: llvm-objdump -s %t2 | FileCheck --check-prefix CHECK-DUPS %s
// RUN: ld.lld %t -o %t3
// RUN: llvm-objdump -s %t3 | FileCheck %s

/// Test that lld can at least remove duplicate .ARM.exidx sections. A more
/// fine grained implementation will be able to remove duplicate entries within
/// a .ARM.exidx section.

/// With duplicate entries
// CHECK-DUPS: Contents of section .ARM.exidx:
// CHECK-DUPS-NEXT: 100d4 78000100 01000000 74000100 01000000
// CHECK-DUPS-NEXT: 100e4 70000100 01000000 6c000100 01000000
// CHECK-DUPS-NEXT: 100f4 68000100 08849780 64000100 08849780
// CHECK-DUPS-NEXT: 10104 60000100 08849780 5c000100 24000000
// CHECK-DUPS-NEXT: 10114 58000100 28000000 54000100 01000000
// CHECK-DUPS-NEXT: 10124 50000100 01000000 4c000100 01000000
// CHECK-DUPS-NEXT: Contents of section .ARM.extab:

/// After duplicate entry removal
// CHECK: Contents of section .ARM.exidx:
// CHECK-NEXT: 100d4 48000100 01000000 50000100 08849780
// CHECK-NEXT: 100e4 54000100 1c000000 50000100 20000000
// CHECK-NEXT: 100f4 4c000100 01000000 4c000100 01000000
// CHECK-NEXT: Contents of section .ARM.extab:
        .syntax unified

        /// Expect 1 EXIDX_CANTUNWIND entry.
        .section .text.00, "ax", %progbits
        .globl _start
_start:
        .fnstart
        bx lr
        .cantunwind
        .fnend

        /// Expect .ARM.exidx.text.01 to be identical to .ARM.exidx.text.00
        .section .text.01, "ax", %progbits
        .globl f1
f1:
        .fnstart
        bx lr
        .cantunwind
        .fnend

        /// Expect 2 EXIDX_CANTUNWIND entries, these can be duplicated into
        /// .ARM.exid.text.00
        .section .text.02, "ax", %progbits
        .globl f2
f2:
        .fnstart
        bx lr
        .cantunwind
        .fnend

        .globl f3
f3:
        .fnstart
        bx lr
        .cantunwind
        .fnend

        /// Expect inline unwind instructions, not a duplicate of previous entry.
        .section .text.03, "ax", %progbits
        .global f4
f4:
        .fnstart
        bx lr
        .save {r7, lr}
        .setfp r7, sp, #0
        .fnend

        /// Expect 2 inline unwind entries that are a duplicate of
        /// .ARM.exidx.text.03
        .section .text.04, "ax", %progbits
        .global f5
f5:
        .fnstart
        bx lr
        .save {r7, lr}
        .setfp r7, sp, #0
        .fnend

        .global f6
f6:
        .fnstart
        bx lr
        .save {r7, lr}
        .setfp r7, sp, #0
        .fnend

        /// Expect a section with a reference to an .ARM.extab. Not a duplicate
        /// of previous inline table entry.
        .section .text.05, "ax",%progbits
        .global f7
f7:
        .fnstart
        bx lr
        .personality __gxx_personality_v0
        .handlerdata
        .long 0
        .fnend

        /// Expect a reference to an identical .ARM.extab. We do not try to
        /// deduplicate references to .ARM.extab sections.
        .section .text.06, "ax",%progbits
        .global f8
f8:
        .fnstart
        bx lr
        .personality __gxx_personality_v0
        .handlerdata
        .long 0
        .fnend

/// Dummy implementation of personality routines to satisfy reference from
/// exception tables
/// Expect Linker generated EXIDX_CANTUNWIND tables
        .section .text.__gcc_personality_v0, "ax", %progbits
        .global __gxx_personality_v0
__gxx_personality_v0:
        bx lr

        .section .text.__aeabi_unwind_cpp_pr0, "ax", %progbits
        .global __aeabi_unwind_cpp_pr0
__aeabi_unwind_cpp_pr0:
        bx lr
