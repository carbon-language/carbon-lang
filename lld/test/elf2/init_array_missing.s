// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s
// REQUIRES: x86

.globl _start
_start:
  call __init_array_start
  call __init_array_end

// With no .init_array section the symbols resolve to 0
// 0 - (0x11000 + 5) = -69637
// 0 - (0x11005 + 5) = -69642

// CHECK: Disassembly of section .text:
// CHECK-NEXT:  _start:
// CHECK-NEXT:   11000:	e8 fb ef fe ff 	callq	-69637
// CHECK-NEXT:   11005:	e8 f6 ef fe ff 	callq	-69642
