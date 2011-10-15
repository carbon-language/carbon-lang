// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: blsmskl  %r11d, %r10d
// CHECK: encoding: [0xc4,0xc2,0x28,0xf3,0xd3]
          blsmskl  %r11d, %r10d

// CHECK: blsmskq  %r11, %r10
// CHECK: encoding: [0xc4,0xc2,0xa8,0xf3,0xd3]
          blsmskq  %r11, %r10

// CHECK: blsmskl  (%rax), %r10d
// CHECK: encoding: [0xc4,0xe2,0x28,0xf3,0x10]
          blsmskl  (%rax), %r10d

// CHECK: blsmskq  (%rax), %r10
// CHECK: encoding: [0xc4,0xe2,0xa8,0xf3,0x10]
          blsmskq  (%rax), %r10

// CHECK: blsil  %r11d, %r10d
// CHECK: encoding: [0xc4,0xc2,0x28,0xf3,0xdb]
          blsil  %r11d, %r10d

// CHECK: blsiq  %r11, %r10
// CHECK: encoding: [0xc4,0xc2,0xa8,0xf3,0xdb]
          blsiq  %r11, %r10

// CHECK: blsil  (%rax), %r10d
// CHECK: encoding: [0xc4,0xe2,0x28,0xf3,0x18]
          blsil  (%rax), %r10d

// CHECK: blsiq  (%rax), %r10
// CHECK: encoding: [0xc4,0xe2,0xa8,0xf3,0x18]
          blsiq  (%rax), %r10

// CHECK: blsrl  %r11d, %r10d
// CHECK: encoding: [0xc4,0xc2,0x28,0xf3,0xcb]
          blsrl  %r11d, %r10d

// CHECK: blsrq  %r11, %r10
// CHECK: encoding: [0xc4,0xc2,0xa8,0xf3,0xcb]
          blsrq  %r11, %r10

// CHECK: blsrl  (%rax), %r10d
// CHECK: encoding: [0xc4,0xe2,0x28,0xf3,0x08]
          blsrl  (%rax), %r10d

// CHECK: blsrq  (%rax), %r10
// CHECK: encoding: [0xc4,0xe2,0xa8,0xf3,0x08]
          blsrq  (%rax), %r10

// CHECK: andnl  (%rax), %r11d, %r10d
// CHECK: encoding: [0xc4,0x62,0x20,0xf2,0x10]
          andnl  (%rax), %r11d, %r10d

// CHECK: andnq  (%rax), %r11, %r10
// CHECK: encoding: [0xc4,0x62,0xa0,0xf2,0x10]
          andnq  (%rax), %r11, %r10
