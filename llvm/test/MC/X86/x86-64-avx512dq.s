// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=skx  --show-encoding %s | FileCheck %s

// CHECK: knotb  %k3, %k3
// CHECK:  encoding: [0xc5,0xf9,0x44,0xdb]
          knotb  %k3, %k3

// CHECK: kmovb  %k3, %k5
// CHECK:  encoding: [0xc5,0xf9,0x90,0xeb]
          kmovb  %k3, %k5

// CHECK: kmovb  (%rcx), %k5
// CHECK:  encoding: [0xc5,0xf9,0x90,0x29]
          kmovb  (%rcx), %k5

// CHECK: kmovb  4660(%rax,%r14,8), %k5
// CHECK:  encoding: [0xc4,0xa1,0x79,0x90,0xac,0xf0,0x34,0x12,0x00,0x00]
          kmovb  4660(%rax,%r14,8), %k5

// CHECK: kmovb  %k2, (%rcx)
// CHECK:  encoding: [0xc5,0xf9,0x91,0x11]
          kmovb  %k2, (%rcx)

// CHECK: kmovb  %k2, 4660(%rax,%r14,8)
// CHECK:  encoding: [0xc4,0xa1,0x79,0x91,0x94,0xf0,0x34,0x12,0x00,0x00]
          kmovb  %k2, 4660(%rax,%r14,8)

// CHECK: kmovb  %eax, %k2
// CHECK:  encoding: [0xc5,0xf9,0x92,0xd0]
          kmovb  %eax, %k2

// CHECK: kmovb  %ebp, %k2
// CHECK:  encoding: [0xc5,0xf9,0x92,0xd5]
          kmovb  %ebp, %k2

// CHECK: kmovb  %r13d, %k2
// CHECK:  encoding: [0xc4,0xc1,0x79,0x92,0xd5]
          kmovb  %r13d, %k2

// CHECK: kmovb  %k3, %eax
// CHECK:  encoding: [0xc5,0xf9,0x93,0xc3]
          kmovb  %k3, %eax

// CHECK: kmovb  %k3, %ebp
// CHECK:  encoding: [0xc5,0xf9,0x93,0xeb]
          kmovb  %k3, %ebp

// CHECK: kmovb  %k3, %r13d
// CHECK:  encoding: [0xc5,0x79,0x93,0xeb]
          kmovb  %k3, %r13d
