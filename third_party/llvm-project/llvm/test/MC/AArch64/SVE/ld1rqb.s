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

ld1rqb  { z0.b }, p0/z, [x0]
// CHECK-INST: ld1rqb  { z0.b }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0x20,0x00,0xa4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 20 00 a4 <unknown>

ld1rqb  { z0.b }, p0/z, [x0, x0]
// CHECK-INST: ld1rqb  { z0.b }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x00,0x00,0xa4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 00 00 a4 <unknown>

ld1rqb  { z31.b }, p7/z, [sp, #-16]
// CHECK-INST: ld1rqb  { z31.b }, p7/z, [sp, #-16]
// CHECK-ENCODING: [0xff,0x3f,0x0f,0xa4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3f 0f a4 <unknown>

ld1rqb  {  z23.b  }, p3/z, [x13, #-128]
// CHECK-INST: ld1rqb  {  z23.b  }, p3/z, [x13, #-128]
// CHECK-ENCODING: [0xb7,0x2d,0x08,0xa4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: b7 2d 08 a4 <unknown>

ld1rqb  {  z21.b  }, p5/z, [x10, #112]
// CHECK-INST: ld1rqb  {  z21.b  }, p5/z, [x10, #112]
// CHECK-ENCODING: [0x55,0x35,0x07,0xa4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 55 35 07 a4 <unknown>
