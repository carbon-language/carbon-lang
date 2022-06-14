// We check that unrecognized opcodes are disassembled by llvm-objdump as data using the .long directive 
// and any trailing bytes are disassembled using the .byte directive
// RUN: llvm-mc -filetype=obj -triple=amdgcn--amdpal -mcpu=gfx900 -show-encoding %s | llvm-objdump -d --mcpu=gfx900 - | FileCheck %s

.text
        v_mov_b32     v7, s24
  v_mov_b32     v8, s25
  .long         0xabadc0de
  s_nop         0
  s_endpgm
  .long         0xabadc0d1, 0xabadc0d2, 0xabadc0d3, 0xabadc0d4
  .byte         0x0a, 0x0b
  .byte         0x0c

// CHECK: .text
// CHECK: v_mov_b32
// CHECK: v_mov_b32
// CHECK: .long 0xabadc0de
// CHECK-SAME: : ABADC0DE
// CHECK: s_endpgm
// CHECK: .long 0xabadc0d1
// CHECK: .long 0xabadc0d2
// CHECK: .long 0xabadc0d3
// CHECK: .long 0xabadc0d4
// CHECK: .byte 0x0a, 0x0b, 0x0c
// CHECK-SAME: : 0A 0B 0C
// CHECK-NOT: .long
