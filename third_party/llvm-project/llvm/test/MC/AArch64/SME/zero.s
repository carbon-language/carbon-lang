// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

zero    {}
// CHECK-INST: zero    {}
// CHECK-ENCODING: [0x00,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 08 c0 <unknown>

zero    {za0.d, za2.d, za4.d, za6.d}
// CHECK-INST: zero {za0.h}
// CHECK-ENCODING: [0x55,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 00 08 c0 <unknown>

zero    {za0.d, za1.d, za2.d, za4.d, za5.d, za7.d}
// CHECK-INST: zero    {za0.d, za1.d, za2.d, za4.d, za5.d, za7.d}
// CHECK-ENCODING: [0xb7,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: b7 00 08 c0 <unknown>

zero    {za0.d, za1.d, za2.d, za3.d, za4.d, za5.d, za6.d, za7.d}
// CHECK-INST: zero {za}
// CHECK-ENCODING: [0xff,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 00 08 c0 <unknown>

// --------------------------------------------------------------------------//
// Aliases

zero {za}
// CHECK-INST: zero {za}
// CHECK-ENCODING: [0xff,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 00 08 c0 <unknown>

zero {za0.b}
// CHECK-INST: zero {za}
// CHECK-ENCODING: [0xff,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 00 08 c0 <unknown>

zero {za0.h}
// CHECK-INST: zero {za0.h}
// CHECK-ENCODING: [0x55,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 00 08 c0 <unknown>

zero {za1.h}
// CHECK-INST: zero {za1.h}
// CHECK-ENCODING: [0xaa,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: aa 00 08 c0 <unknown>

zero {za0.h,za1.h}
// CHECK-INST: zero {za}
// CHECK-ENCODING: [0xff,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 00 08 c0 <unknown>

zero {za0.s}
// CHECK-INST: zero {za0.s}
// CHECK-ENCODING: [0x11,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 11 00 08 c0 <unknown>

zero {za1.s}
// CHECK-INST: zero {za1.s}
// CHECK-ENCODING: [0x22,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 00 08 c0 <unknown>

zero {za2.s}
// CHECK-INST: zero {za2.s}
// CHECK-ENCODING: [0x44,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 44 00 08 c0 <unknown>

zero {za3.s}
// CHECK-INST: zero {za3.s}
// CHECK-ENCODING: [0x88,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 88 00 08 c0 <unknown>

zero {za0.s,za1.s}
// CHECK-INST: zero {za0.s,za1.s}
// CHECK-ENCODING: [0x33,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 33 00 08 c0 <unknown>

zero {za0.s,za2.s}
// CHECK-INST: zero {za0.h}
// CHECK-ENCODING: [0x55,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 00 08 c0 <unknown>

zero {za0.s,za3.s}
// CHECK-INST: zero {za0.s,za3.s}
// CHECK-ENCODING: [0x99,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 99 00 08 c0 <unknown>

zero {za1.s,za2.s}
// CHECK-INST: zero {za1.s,za2.s}
// CHECK-ENCODING: [0x66,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 66 00 08 c0 <unknown>

zero {za1.s,za3.s}
// CHECK-INST: zero {za1.h}
// CHECK-ENCODING: [0xaa,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: aa 00 08 c0 <unknown>

zero {za2.s,za3.s}
// CHECK-INST: zero {za2.s,za3.s}
// CHECK-ENCODING: [0xcc,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cc 00 08 c0 <unknown>

zero {za0.s,za1.s,za2.s}
// CHECK-INST: zero {za0.s,za1.s,za2.s}
// CHECK-ENCODING: [0x77,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 77 00 08 c0 <unknown>

zero {za0.s,za1.s,za3.s}
// CHECK-INST: zero {za0.s,za1.s,za3.s}
// CHECK-ENCODING: [0xbb,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: bb 00 08 c0 <unknown>

zero {za0.s,za2.s,za3.s}
// CHECK-INST: zero {za0.s,za2.s,za3.s}
// CHECK-ENCODING: [0xdd,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 00 08 c0 <unknown>

zero {za1.s,za2.s,za3.s}
// CHECK-INST: zero {za1.s,za2.s,za3.s}
// CHECK-ENCODING: [0xee,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ee 00 08 c0 <unknown>

zero {za0.s,za1.s,za2.s,za3.s}
// CHECK-INST: zero {za}
// CHECK-ENCODING: [0xff,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 00 08 c0 <unknown>

zero {za0.d,za1.d,za2.d,za3.d,za4.d,za5.d,za6.d,za7.d}
// CHECK-INST: zero {za}
// CHECK-ENCODING: [0xff,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 00 08 c0 <unknown>

zero {za0.d,za2.d,za4.d,za6.d}
// CHECK-INST: zero {za0.h}
// CHECK-ENCODING: [0x55,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 55 00 08 c0 <unknown>

zero {za1.d,za3.d,za5.d,za7.d}
// CHECK-INST: zero {za1.h}
// CHECK-ENCODING: [0xaa,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: aa 00 08 c0 <unknown>

zero {za0.d,za4.d}
// CHECK-INST: zero {za0.s}
// CHECK-ENCODING: [0x11,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 11 00 08 c0 <unknown>

zero {za1.d,za5.d}
// CHECK-INST: zero {za1.s}
// CHECK-ENCODING: [0x22,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 00 08 c0 <unknown>

zero {za2.d,za6.d}
// CHECK-INST: zero {za2.s}
// CHECK-ENCODING: [0x44,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 44 00 08 c0 <unknown>

zero {za3.d,za7.d}
// CHECK-INST: zero {za3.s}
// CHECK-ENCODING: [0x88,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 88 00 08 c0 <unknown>

zero {za0.d,za1.d,za4.d,za5.d}
// CHECK-INST: zero {za0.s,za1.s}
// CHECK-ENCODING: [0x33,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 33 00 08 c0 <unknown>

zero {za0.d,za3.d,za4.d,za7.d}
// CHECK-INST: zero {za0.s,za3.s}
// CHECK-ENCODING: [0x99,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 99 00 08 c0 <unknown>

zero {za1.d,za2.d,za5.d,za6.d}
// CHECK-INST: zero {za1.s,za2.s}
// CHECK-ENCODING: [0x66,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 66 00 08 c0 <unknown>

zero {za2.d,za3.d,za6.d,za7.d}
// CHECK-INST: zero {za2.s,za3.s}
// CHECK-ENCODING: [0xcc,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cc 00 08 c0 <unknown>

zero {za0.d,za1.d,za2.d,za4.d,za5.d,za6.d}
// CHECK-INST: zero {za0.s,za1.s,za2.s}
// CHECK-ENCODING: [0x77,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 77 00 08 c0 <unknown>

zero {za0.d,za1.d,za3.d,za4.d,za5.d,za7.d}
// CHECK-INST: zero {za0.s,za1.s,za3.s}
// CHECK-ENCODING: [0xbb,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: bb 00 08 c0 <unknown>

zero {za0.d,za2.d,za3.d,za4.d,za6.d,za7.d}
// CHECK-INST: zero {za0.s,za2.s,za3.s}
// CHECK-ENCODING: [0xdd,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: dd 00 08 c0 <unknown>

zero {za1.d,za2.d,za3.d,za5.d,za6.d,za7.d}
// CHECK-INST: zero {za1.s,za2.s,za3.s}
// CHECK-ENCODING: [0xee,0x00,0x08,0xc0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ee 00 08 c0 <unknown>
