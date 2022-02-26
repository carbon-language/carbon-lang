// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


uqxtnb z0.b, z31.h
// CHECK-INST: uqxtnb	z0.b, z31.h
// CHECK-ENCODING: [0xe0,0x4b,0x28,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e0 4b 28 45 <unknown>

uqxtnb z0.h, z31.s
// CHECK-INST: uqxtnb	z0.h, z31.s
// CHECK-ENCODING: [0xe0,0x4b,0x30,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e0 4b 30 45 <unknown>

uqxtnb z0.s, z31.d
// CHECK-INST: uqxtnb	z0.s, z31.d
// CHECK-ENCODING: [0xe0,0x4b,0x60,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e0 4b 60 45 <unknown>
