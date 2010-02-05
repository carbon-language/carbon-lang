// RUN: llvm-mc -triple i386-unknown-unknown %s -show-encoding | FileCheck %s

fisttpl	3735928559(%ebx,%ecx,8)

# CHECK: encoding: [0xdb,0x8c,0xcb,0xef,0xbe,0xad,0xde]
