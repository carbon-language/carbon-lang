// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

compact z31.s, p7, z31.s
// CHECK-INST: compact z31.s, p7, z31.s
// CHECK-ENCODING: [0xff,0x9f,0xa1,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f a1 05 <unknown>

compact z31.d, p7, z31.d
// CHECK-INST: compact z31.d, p7, z31.d
// CHECK-ENCODING: [0xff,0x9f,0xe1,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f e1 05 <unknown>
