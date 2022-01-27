// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2-aes < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2-aes < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2-aes - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2-aes < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


pmullb z29.q, z30.d, z31.d
// CHECK-INST: pmullb z29.q, z30.d, z31.d
// CHECK-ENCODING: [0xdd,0x6b,0x1f,0x45]
// CHECK-ERROR: instruction requires: sve2-aes
// CHECK-UNKNOWN: dd 6b 1f 45 <unknown>
