// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

rbit  z0.b, p7/m, z31.b
// CHECK-INST: rbit	z0.b, p7/m, z31.b
// CHECK-ENCODING: [0xe0,0x9f,0x27,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 9f 27 05 <unknown>

rbit  z0.h, p7/m, z31.h
// CHECK-INST: rbit	z0.h, p7/m, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x67,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 9f 67 05 <unknown>

rbit  z0.s, p7/m, z31.s
// CHECK-INST: rbit	z0.s, p7/m, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0xa7,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 9f a7 05 <unknown>

rbit  z0.d, p7/m, z31.d
// CHECK-INST: rbit	z0.d, p7/m, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xe7,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 9f e7 05 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0.d, p7/z, z7.d
// CHECK-INST: movprfx	z0.d, p7/z, z7.d
// CHECK-ENCODING: [0xe0,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 3c d0 04 <unknown>

rbit  z0.d, p7/m, z31.d
// CHECK-INST: rbit	z0.d, p7/m, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xe7,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 9f e7 05 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

rbit  z0.d, p7/m, z31.d
// CHECK-INST: rbit	z0.d, p7/m, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xe7,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 9f e7 05 <unknown>
