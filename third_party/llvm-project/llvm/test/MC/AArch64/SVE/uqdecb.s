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


// ---------------------------------------------------------------------------//
// Test 64-bit form (x0) and its aliases
// ---------------------------------------------------------------------------//
uqdecb  x0
// CHECK-INST: uqdecb  x0
// CHECK-ENCODING: [0xe0,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff 30 04 <unknown>

uqdecb  x0, all
// CHECK-INST: uqdecb  x0
// CHECK-ENCODING: [0xe0,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff 30 04 <unknown>

uqdecb  x0, all, mul #1
// CHECK-INST: uqdecb  x0
// CHECK-ENCODING: [0xe0,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff 30 04 <unknown>

uqdecb  x0, all, mul #16
// CHECK-INST: uqdecb  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xff,0x3f,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff 3f 04 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (w0) and its aliases
// ---------------------------------------------------------------------------//

uqdecb  w0
// CHECK-INST: uqdecb  w0
// CHECK-ENCODING: [0xe0,0xff,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff 20 04 <unknown>

uqdecb  w0, all
// CHECK-INST: uqdecb  w0
// CHECK-ENCODING: [0xe0,0xff,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff 20 04 <unknown>

uqdecb  w0, all, mul #1
// CHECK-INST: uqdecb  w0
// CHECK-ENCODING: [0xe0,0xff,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff 20 04 <unknown>

uqdecb  w0, all, mul #16
// CHECK-INST: uqdecb  w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xff,0x2f,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff 2f 04 <unknown>

uqdecb  w0, pow2
// CHECK-INST: uqdecb  w0, pow2
// CHECK-ENCODING: [0x00,0xfc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fc 20 04 <unknown>

uqdecb  w0, pow2, mul #16
// CHECK-INST: uqdecb  w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xfc,0x2f,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fc 2f 04 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

uqdecb  x0, pow2
// CHECK-INST: uqdecb  x0, pow2
// CHECK-ENCODING: [0x00,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fc 30 04 <unknown>

uqdecb  x0, vl1
// CHECK-INST: uqdecb  x0, vl1
// CHECK-ENCODING: [0x20,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 fc 30 04 <unknown>

uqdecb  x0, vl2
// CHECK-INST: uqdecb  x0, vl2
// CHECK-ENCODING: [0x40,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 fc 30 04 <unknown>

uqdecb  x0, vl3
// CHECK-INST: uqdecb  x0, vl3
// CHECK-ENCODING: [0x60,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 fc 30 04 <unknown>

uqdecb  x0, vl4
// CHECK-INST: uqdecb  x0, vl4
// CHECK-ENCODING: [0x80,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 fc 30 04 <unknown>

uqdecb  x0, vl5
// CHECK-INST: uqdecb  x0, vl5
// CHECK-ENCODING: [0xa0,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 fc 30 04 <unknown>

uqdecb  x0, vl6
// CHECK-INST: uqdecb  x0, vl6
// CHECK-ENCODING: [0xc0,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 fc 30 04 <unknown>

uqdecb  x0, vl7
// CHECK-INST: uqdecb  x0, vl7
// CHECK-ENCODING: [0xe0,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fc 30 04 <unknown>

uqdecb  x0, vl8
// CHECK-INST: uqdecb  x0, vl8
// CHECK-ENCODING: [0x00,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fd 30 04 <unknown>

uqdecb  x0, vl16
// CHECK-INST: uqdecb  x0, vl16
// CHECK-ENCODING: [0x20,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 fd 30 04 <unknown>

uqdecb  x0, vl32
// CHECK-INST: uqdecb  x0, vl32
// CHECK-ENCODING: [0x40,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 fd 30 04 <unknown>

uqdecb  x0, vl64
// CHECK-INST: uqdecb  x0, vl64
// CHECK-ENCODING: [0x60,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 fd 30 04 <unknown>

uqdecb  x0, vl128
// CHECK-INST: uqdecb  x0, vl128
// CHECK-ENCODING: [0x80,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 fd 30 04 <unknown>

uqdecb  x0, vl256
// CHECK-INST: uqdecb  x0, vl256
// CHECK-ENCODING: [0xa0,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 fd 30 04 <unknown>

uqdecb  x0, #14
// CHECK-INST: uqdecb  x0, #14
// CHECK-ENCODING: [0xc0,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 fd 30 04 <unknown>

uqdecb  x0, #15
// CHECK-INST: uqdecb  x0, #15
// CHECK-ENCODING: [0xe0,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fd 30 04 <unknown>

uqdecb  x0, #16
// CHECK-INST: uqdecb  x0, #16
// CHECK-ENCODING: [0x00,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fe 30 04 <unknown>

uqdecb  x0, #17
// CHECK-INST: uqdecb  x0, #17
// CHECK-ENCODING: [0x20,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 fe 30 04 <unknown>

uqdecb  x0, #18
// CHECK-INST: uqdecb  x0, #18
// CHECK-ENCODING: [0x40,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 fe 30 04 <unknown>

uqdecb  x0, #19
// CHECK-INST: uqdecb  x0, #19
// CHECK-ENCODING: [0x60,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 fe 30 04 <unknown>

uqdecb  x0, #20
// CHECK-INST: uqdecb  x0, #20
// CHECK-ENCODING: [0x80,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 fe 30 04 <unknown>

uqdecb  x0, #21
// CHECK-INST: uqdecb  x0, #21
// CHECK-ENCODING: [0xa0,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 fe 30 04 <unknown>

uqdecb  x0, #22
// CHECK-INST: uqdecb  x0, #22
// CHECK-ENCODING: [0xc0,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 fe 30 04 <unknown>

uqdecb  x0, #23
// CHECK-INST: uqdecb  x0, #23
// CHECK-ENCODING: [0xe0,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fe 30 04 <unknown>

uqdecb  x0, #24
// CHECK-INST: uqdecb  x0, #24
// CHECK-ENCODING: [0x00,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 ff 30 04 <unknown>

uqdecb  x0, #25
// CHECK-INST: uqdecb  x0, #25
// CHECK-ENCODING: [0x20,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 ff 30 04 <unknown>

uqdecb  x0, #26
// CHECK-INST: uqdecb  x0, #26
// CHECK-ENCODING: [0x40,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 ff 30 04 <unknown>

uqdecb  x0, #27
// CHECK-INST: uqdecb  x0, #27
// CHECK-ENCODING: [0x60,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 ff 30 04 <unknown>

uqdecb  x0, #28
// CHECK-INST: uqdecb  x0, #28
// CHECK-ENCODING: [0x80,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 ff 30 04 <unknown>
