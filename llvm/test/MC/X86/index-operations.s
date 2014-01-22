// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=64 %s
// RUN: FileCheck --check-prefix=ERR64 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=32 %s
// RUN: FileCheck --check-prefix=ERR32 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown-code16 --show-encoding %s 2> %t.err | FileCheck --check-prefix=16 %s
// RUN: FileCheck --check-prefix=ERR16 < %t.err %s

lodsb
// 64: lodsb (%rsi), %al # encoding: [0xac]
// 32: lodsb (%esi), %al # encoding: [0xac]
// 16: lodsb (%si), %al # encoding: [0xac]

lodsb (%rsi), %al
// 64: lodsb (%rsi), %al # encoding: [0xac]
// ERR32: 64-bit
// ERR16: 64-bit

lodsb (%esi), %al
// 64: lodsb (%esi), %al # encoding: [0x67,0xac]
// 32: lodsb (%esi), %al # encoding: [0xac]
// 16: lodsb (%esi), %al # encoding: [0x67,0xac]

lodsb (%si), %al
// ERR64: invalid 16-bit base register
// 32: lodsb (%si), %al # encoding: [0x67,0xac]
// 16: lodsb (%si), %al # encoding: [0xac]

lodsl %gs:(%esi)
// 64: lodsl %gs:(%esi), %eax # encoding: [0x65,0x67,0xad]
// 32: lodsl %gs:(%esi), %eax # encoding: [0x65,0xad]
// 16: lodsl %gs:(%esi), %eax # encoding: [0x66,0x65,0x67,0xad]

lodsl (%edi), %eax
// ERR64: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

lodsl 44(%edi), %eax
// ERR64: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

lods (%esi), %ax
// 64: lodsw (%esi), %ax # encoding: [0x66,0x67,0xad]
// 32: lodsw (%esi), %ax # encoding: [0x66,0xad]
// 16: lodsw (%esi), %ax # encoding: [0x67,0xad]
