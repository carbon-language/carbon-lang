// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+bf16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve,+bf16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

bfcvtnt z0.H, p0/m, z1.S
// CHECK-INST: bfcvtnt z0.h, p0/m, z1.s
// CHECK-ENCODING: [0x20,0xa0,0x8a,0x64]
// CHECK-ERROR: instruction requires: bf16 sve or sme

movprfx z0.S, p0/m, z2.S
// CHECK-INST: movprfx z0.s, p0/m, z2.s
// CHECK-ENCODING: [0x40,0x20,0x91,0x04]
// CHECK-ERROR: instruction requires: sve or sme

bfcvtnt z0.H, p0/m, z1.S
// CHECK-INST: bfcvtnt z0.h, p0/m, z1.s
// CHECK-ENCODING: [0x20,0xa0,0x8a,0x64]
// CHECK-ERROR: instruction requires: bf16 sve or sme

movprfx z0, z2
// CHECK-INST: movprfx z0, z2
// CHECK-ENCODING: [0x40,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme

bfcvtnt z0.H, p0/m, z1.S
// CHECK-INST: bfcvtnt z0.h, p0/m, z1.s
// CHECK-ENCODING: [0x20,0xa0,0x8a,0x64]
// CHECK-ERROR: instruction requires: bf16 sve or sme
