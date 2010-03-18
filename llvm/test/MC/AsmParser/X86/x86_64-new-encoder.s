// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

movl	foo(%rip), %eax
// CHECK: movl	foo(%rip), %eax
// CHECK: encoding: [0x8b,0x05,A,A,A,A]
// CHECK: fixup A - offset: 2, value: foo-4, kind: reloc_riprel_4byte

movb	$12, foo(%rip)
// CHECK: movb	$12, foo(%rip)
// CHECK: encoding: [0xc6,0x05,A,A,A,A,0x0c]
// CHECK:    fixup A - offset: 2, value: foo-5, kind: reloc_riprel_4byte

movw	$12, foo(%rip)
// CHECK: movw	$12, foo(%rip)
// CHECK: encoding: [0x66,0xc7,0x05,A,A,A,A,0x0c,0x00]
// CHECK:    fixup A - offset: 3, value: foo-6, kind: reloc_riprel_4byte

movl	$12, foo(%rip)
// CHECK: movl	$12, foo(%rip)
// CHECK: encoding: [0xc7,0x05,A,A,A,A,0x0c,0x00,0x00,0x00]
// CHECK:    fixup A - offset: 2, value: foo-8, kind: reloc_riprel_4byte

movq	$12, foo(%rip)
// CHECK:  movq	$12, foo(%rip)
// CHECK: encoding: [0x48,0xc7,0x05,A,A,A,A,0x0c,0x00,0x00,0x00]
// CHECK:    fixup A - offset: 3, value: foo-8, kind: reloc_riprel_4byte

// CHECK: addq	$-424, %rax
// CHECK: encoding: [0x48,0x05,0x58,0xfe,0xff,0xff]
addq $-424, %rax


// CHECK: movq	_foo@GOTPCREL(%rip), %rax
// CHECK:  encoding: [0x48,0x8b,0x05,A,A,A,A]
// CHECK:  fixup A - offset: 3, value: _foo@GOTPCREL, kind: reloc_riprel_4byte_movq_load
movq _foo@GOTPCREL(%rip), %rax


// CHECK: movq	(%r13,%rax,8), %r13
// CHECK:  encoding: [0x4d,0x8b,0x6c,0xc5,0x00]
movq 0x00(%r13,%rax,8),%r13
