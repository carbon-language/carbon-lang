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

sqabs z31.b, p7/m, z31.b
// CHECK-INST: sqabs z31.b, p7/m, z31.b
// CHECK-ENCODING: [0xff,0xbf,0x08,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff bf 08 44 <unknown>

sqabs z31.h, p7/m, z31.h
// CHECK-INST: sqabs z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x48,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff bf 48 44 <unknown>

sqabs z31.s, p7/m, z31.s
// CHECK-INST: sqabs z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x88,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff bf 88 44 <unknown>

sqabs z31.d, p7/m, z31.d
// CHECK-INST: sqabs z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc8,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff bf c8 44 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.s, p7/z, z6.s
// CHECK-INST: movprfx	z4.s, p7/z, z6.s
// CHECK-ENCODING: [0xc4,0x3c,0x90,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 3c 90 04 <unknown>

sqabs z4.s, p7/m, z31.s
// CHECK-INST: sqabs z4.s, p7/m, z31.s
// CHECK-ENCODING: [0xe4,0xbf,0x88,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e4 bf 88 44 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

sqabs z4.s, p7/m, z31.s
// CHECK-INST: sqabs z4.s, p7/m, z31.s
// CHECK-ENCODING: [0xe4,0xbf,0x88,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: e4 bf 88 44 <unknown>
