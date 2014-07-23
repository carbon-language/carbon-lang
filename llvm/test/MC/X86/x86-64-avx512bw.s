// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=skx  --show-encoding %s | FileCheck %s

// CHECK: knotq  %k6, %k3
// CHECK:  encoding: [0xc4,0xe1,0xf8,0x44,0xde]
          knotq  %k6, %k3

// CHECK: knotd  %k4, %k3
// CHECK:  encoding: [0xc4,0xe1,0xf9,0x44,0xdc]
          knotd  %k4, %k3

// CHECK: kmovq  %k5, %k2
// CHECK:  encoding: [0xc4,0xe1,0xf8,0x90,0xd5]
          kmovq  %k5, %k2

// CHECK: kmovq  (%rcx), %k2
// CHECK:  encoding: [0xc4,0xe1,0xf8,0x90,0x11]
          kmovq  (%rcx), %k2

// CHECK: kmovq  291(%rax,%r14,8), %k2
// CHECK:  encoding: [0xc4,0xa1,0xf8,0x90,0x94,0xf0,0x23,0x01,0x00,0x00]
          kmovq  291(%rax,%r14,8), %k2

// CHECK: kmovd  %k4, %k5
// CHECK:  encoding: [0xc4,0xe1,0xf9,0x90,0xec]
          kmovd  %k4, %k5

// CHECK: kmovd  (%rcx), %k5
// CHECK:  encoding: [0xc4,0xe1,0xf9,0x90,0x29]
          kmovd  (%rcx), %k5

// CHECK: kmovd  291(%rax,%r14,8), %k5
// CHECK:  encoding: [0xc4,0xa1,0xf9,0x90,0xac,0xf0,0x23,0x01,0x00,0x00]
          kmovd  291(%rax,%r14,8), %k5

// CHECK: kmovq  %k3, (%rcx)
// CHECK:  encoding: [0xc4,0xe1,0xf8,0x91,0x19]
          kmovq  %k3, (%rcx)

// CHECK: kmovq  %k3, 291(%rax,%r14,8)
// CHECK:  encoding: [0xc4,0xa1,0xf8,0x91,0x9c,0xf0,0x23,0x01,0x00,0x00]
          kmovq  %k3, 291(%rax,%r14,8)

// CHECK: kmovd  %k3, (%rcx)
// CHECK:  encoding: [0xc4,0xe1,0xf9,0x91,0x19]
          kmovd  %k3, (%rcx)

// CHECK: kmovd  %k3, 291(%rax,%r14,8)
// CHECK:  encoding: [0xc4,0xa1,0xf9,0x91,0x9c,0xf0,0x23,0x01,0x00,0x00]
          kmovd  %k3, 291(%rax,%r14,8)

// CHECK: kmovq  %rax, %k2
// CHECK:  encoding: [0xc4,0xe1,0xfb,0x92,0xd0]
          kmovq  %rax, %k2

// CHECK: kmovq  %r8, %k2
// CHECK:  encoding: [0xc4,0xc1,0xfb,0x92,0xd0]
          kmovq  %r8, %k2

// CHECK: kmovd  %eax, %k4
// CHECK:  encoding: [0xc5,0xfb,0x92,0xe0]
          kmovd  %eax, %k4

// CHECK: kmovd  %ebp, %k4
// CHECK:  encoding: [0xc5,0xfb,0x92,0xe5]
          kmovd  %ebp, %k4

// CHECK: kmovd  %r13d, %k4
// CHECK:  encoding: [0xc4,0xc1,0x7b,0x92,0xe5]
          kmovd  %r13d, %k4

// CHECK: kmovq  %k3, %rax
// CHECK:  encoding: [0xc4,0xe1,0xfb,0x93,0xc3]
          kmovq  %k3, %rax

// CHECK: kmovq  %k3, %r8
// CHECK:  encoding: [0xc4,0x61,0xfb,0x93,0xc3]
          kmovq  %k3, %r8

// CHECK: kmovd  %k5, %eax
// CHECK:  encoding: [0xc5,0xfb,0x93,0xc5]
          kmovd  %k5, %eax

// CHECK: kmovd  %k5, %ebp
// CHECK:  encoding: [0xc5,0xfb,0x93,0xed]
          kmovd  %k5, %ebp

// CHECK: kmovd  %k5, %r13d
// CHECK:  encoding: [0xc5,0x7b,0x93,0xed]
          kmovd  %k5, %r13d
