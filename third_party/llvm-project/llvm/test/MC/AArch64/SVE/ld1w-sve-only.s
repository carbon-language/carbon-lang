// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

ld1w    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ld1w    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x40,0x00,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 00 85 <unknown>

ld1w    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ld1w    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x40,0x40,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 40 85 <unknown>

ld1w    { z31.s }, p7/z, [sp, z31.s, uxtw #2]
// CHECK-INST: ld1w    { z31.s }, p7/z, [sp, z31.s, uxtw #2]
// CHECK-ENCODING: [0xff,0x5f,0x3f,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 5f 3f 85 <unknown>

ld1w    { z31.s }, p7/z, [sp, z31.s, sxtw #2]
// CHECK-INST: ld1w    { z31.s }, p7/z, [sp, z31.s, sxtw #2]
// CHECK-ENCODING: [0xff,0x5f,0x7f,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 5f 7f 85 <unknown>

ld1w    { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ld1w    { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xdf,0x5f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df 5f c5 <unknown>

ld1w    { z23.d }, p3/z, [x13, z8.d, lsl #2]
// CHECK-INST: ld1w    { z23.d }, p3/z, [x13, z8.d, lsl #2]
// CHECK-ENCODING: [0xb7,0xcd,0x68,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 cd 68 c5 <unknown>

ld1w    { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ld1w    { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x55,0x15,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 15 c5 <unknown>

ld1w    { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ld1w    { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x55,0x55,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 55 c5 <unknown>

ld1w    { z0.d }, p0/z, [x0, z0.d, uxtw #2]
// CHECK-INST: ld1w    { z0.d }, p0/z, [x0, z0.d, uxtw #2]
// CHECK-ENCODING: [0x00,0x40,0x20,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 20 c5 <unknown>

ld1w    { z0.d }, p0/z, [x0, z0.d, sxtw #2]
// CHECK-INST: ld1w    { z0.d }, p0/z, [x0, z0.d, sxtw #2]
// CHECK-ENCODING: [0x00,0x40,0x60,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 60 c5 <unknown>

ld1w    { z31.s }, p7/z, [z31.s, #124]
// CHECK-INST: ld1w    { z31.s }, p7/z, [z31.s, #124]
// CHECK-ENCODING: [0xff,0xdf,0x3f,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df 3f 85 <unknown>

ld1w    { z0.s }, p0/z, [z0.s]
// CHECK-INST: ld1w    { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0xc0,0x20,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 20 85 <unknown>

ld1w    { z31.d }, p7/z, [z31.d, #124]
// CHECK-INST: ld1w    { z31.d }, p7/z, [z31.d, #124]
// CHECK-ENCODING: [0xff,0xdf,0x3f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df 3f c5 <unknown>

ld1w    { z0.d }, p0/z, [z0.d]
// CHECK-INST: ld1w    { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xc0,0x20,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 20 c5 <unknown>
