// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

addspl   x21, x21, #0
// CHECK-INST: addspl   x21, x21, #0
// CHECK-ENCODING: [0x15,0x58,0x75,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 15 58 75 04 <unknown>

addspl   x23, x8, #-1
// CHECK-INST: addspl   x23, x8, #-1
// CHECK-ENCODING: [0xf7,0x5f,0x68,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: f7 5f 68 04 <unknown>

addspl   sp, sp, #31
// CHECK-INST: addspl   sp, sp, #31
// CHECK-ENCODING: [0xff,0x5b,0x7f,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ff 5b 7f 04 <unknown>

addspl   x0, x0, #-32
// CHECK-INST: addspl   x0, x0, #-32
// CHECK-ENCODING: [0x00,0x5c,0x60,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 5c 60 04 <unknown>
