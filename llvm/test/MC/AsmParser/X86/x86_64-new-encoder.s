// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding --enable-new-x86-encoder %s | FileCheck %s

movl	foo(%rip), %eax
// CHECK: movl	foo(%rip), %eax
// CHECK: encoding: [0x8b,0x05,A,A,A,A]
// CHECK: fixup A - offset: 2, value: foo, kind: reloc_riprel_4byte

movb	$12, foo(%rip)
// CHECK: movb	$12, foo(%rip)
// CHECK: encoding: [0xc6,0x05,A,A,A,A,0x0c]
// CHECK:    fixup A - offset: 2, value: foo-1, kind: reloc_riprel_4byte

movw	$12, foo(%rip)
// CHECK: movw	$12, foo(%rip)
// CHECK: encoding: [0x66,0xc7,0x05,A,A,A,A,0x0c,0x00]
// CHECK:    fixup A - offset: 3, value: foo-2, kind: reloc_riprel_4byte

movl	$12, foo(%rip)
// CHECK: movl	$12, foo(%rip)
// CHECK: encoding: [0xc7,0x05,A,A,A,A,0x0c,0x00,0x00,0x00]
// CHECK:    fixup A - offset: 2, value: foo-4, kind: reloc_riprel_4byte

movq	$12, foo(%rip)
// CHECK:  movq	$12, foo(%rip)
// CHECK: encoding: [0x48,0xc7,0x05,A,A,A,A,0x0c,0x00,0x00,0x00]
// CHECK:    fixup A - offset: 3, value: foo-4, kind: reloc_riprel_4byte
