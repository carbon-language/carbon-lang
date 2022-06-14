// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


uabdlb z0.h, z1.b, z2.b
// CHECK-INST: uabdlb z0.h, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x38,0x42,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 38 42 45 <unknown>

uabdlb z29.s, z30.h, z31.h
// CHECK-INST: uabdlb z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0x3b,0x9f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 3b 9f 45 <unknown>

uabdlb z31.d, z31.s, z31.s
// CHECK-INST: uabdlb z31.d, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x3b,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 3b df 45 <unknown>
