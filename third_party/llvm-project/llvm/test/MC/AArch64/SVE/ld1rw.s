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

ld1rw   { z0.s }, p0/z, [x0]
// CHECK-INST: ld1rw   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xc0,0x40,0x85]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 c0 40 85 <unknown>

ld1rw   { z0.d }, p0/z, [x0]
// CHECK-INST: ld1rw   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x40,0x85]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 e0 40 85 <unknown>

ld1rw   { z31.s }, p7/z, [sp, #252]
// CHECK-INST: ld1rw   { z31.s }, p7/z, [sp, #252]
// CHECK-ENCODING: [0xff,0xdf,0x7f,0x85]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff df 7f 85 <unknown>

ld1rw   { z31.d }, p7/z, [sp, #252]
// CHECK-INST: ld1rw   { z31.d }, p7/z, [sp, #252]
// CHECK-ENCODING: [0xff,0xff,0x7f,0x85]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff ff 7f 85 <unknown>
