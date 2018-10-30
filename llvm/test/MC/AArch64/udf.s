# RUN: llvm-mc -assemble -show-encoding -triple=aarch64- %s | FileCheck %s
# CHECK:  .text
# CHECK-NEXT: udf #0      // encoding: [0x00,0x00,0x00,0x00]
# CHECK-NEXT: udf #1      // encoding: [0x01,0x00,0x00,0x00]
# CHECK-NEXT: udf #16     // encoding: [0x10,0x00,0x00,0x00]
# CHECK-NEXT: udf #32     // encoding: [0x20,0x00,0x00,0x00]
# CHECK-NEXT: udf #48     // encoding: [0x30,0x00,0x00,0x00]
# CHECK-NEXT: udf #65535      // encoding: [0xff,0xff,0x00,0x00]
.text
udf 0
udf 1
udf 16
udf 32
udf 48
udf 65535
