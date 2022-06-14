// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

brkpa  p0.b,  p15/z, p1.b,  p2.b
// CHECK-INST: brkpa	p0.b, p15/z, p1.b, p2.b
// CHECK-ENCODING: [0x20,0xfc,0x02,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 fc 02 25 <unknown>

brkpa  p15.b, p15/z, p15.b, p15.b
// CHECK-INST: brkpa	p15.b, p15/z, p15.b, p15.b
// CHECK-ENCODING: [0xef,0xfd,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ef fd 0f 25 <unknown>
