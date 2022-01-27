// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

stnt1w z0.s, p0, [z1.s]
// CHECK-INST: stnt1w { z0.s }, p0, [z1.s]
// CHECK-ENCODING: [0x20,0x20,0x5f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 20 5f e5 <unknown>

stnt1w z31.s, p7, [z31.s, xzr]
// CHECK-INST: stnt1w { z31.s }, p7, [z31.s]
// CHECK-ENCODING: [0xff,0x3f,0x5f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 5f e5 <unknown>

stnt1w z31.s, p7, [z31.s, x0]
// CHECK-INST: stnt1w { z31.s }, p7, [z31.s, x0]
// CHECK-ENCODING: [0xff,0x3f,0x40,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 40 e5 <unknown>

stnt1w z0.d, p0, [z1.d]
// CHECK-INST: stnt1w { z0.d }, p0, [z1.d]
// CHECK-ENCODING: [0x20,0x20,0x1f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 20 1f e5 <unknown>

stnt1w z31.d, p7, [z31.d, xzr]
// CHECK-INST: stnt1w { z31.d }, p7, [z31.d]
// CHECK-ENCODING: [0xff,0x3f,0x1f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 1f e5 <unknown>

stnt1w z31.d, p7, [z31.d, x0]
// CHECK-INST: stnt1w { z31.d }, p7, [z31.d, x0]
// CHECK-ENCODING: [0xff,0x3f,0x00,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 00 e5 <unknown>

stnt1w { z0.s }, p0, [z1.s]
// CHECK-INST: stnt1w { z0.s }, p0, [z1.s]
// CHECK-ENCODING: [0x20,0x20,0x5f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 20 5f e5 <unknown>

stnt1w { z31.s }, p7, [z31.s, xzr]
// CHECK-INST: stnt1w { z31.s }, p7, [z31.s]
// CHECK-ENCODING: [0xff,0x3f,0x5f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 5f e5 <unknown>

stnt1w { z31.s }, p7, [z31.s, x0]
// CHECK-INST: stnt1w { z31.s }, p7, [z31.s, x0]
// CHECK-ENCODING: [0xff,0x3f,0x40,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 40 e5 <unknown>

stnt1w { z0.d }, p0, [z1.d]
// CHECK-INST: stnt1w { z0.d }, p0, [z1.d]
// CHECK-ENCODING: [0x20,0x20,0x1f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 20 1f e5 <unknown>

stnt1w { z31.d }, p7, [z31.d, xzr]
// CHECK-INST: stnt1w { z31.d }, p7, [z31.d]
// CHECK-ENCODING: [0xff,0x3f,0x1f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 1f e5 <unknown>

stnt1w { z31.d }, p7, [z31.d, x0]
// CHECK-INST: stnt1w { z31.d }, p7, [z31.d, x0]
// CHECK-ENCODING: [0xff,0x3f,0x00,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 00 e5 <unknown>
