// RUN: llvm-mc -target-abi=ilp32 -triple=arm64-linux-gnu -show-encoding -o - \
// RUN:   %s \
// RUN:   | FileCheck %s
// RUN: llvm-mc -target-abi=ilp32 -triple=arm64-linux-gnu -show-encoding \
// RUN:   -filetype=obj -o - %s \
// RUN:   | llvm-objdump -r - \
// RUN:   | FileCheck --check-prefix=CHECK-OBJ %s

        movz x2, #:abs_g0:sym
        movk w3, #:abs_g0_nc:sym
        movz x13, #:abs_g0_s:sym
        movn x17, #:abs_g0_s:sym
// CHECK:   movz x2, #:abs_g0:sym // encoding: [0bAAA00010,A,0b100AAAAA,0xd2]
// CHECK-NEXT: // fixup A - offset: 0, value: :abs_g0:sym, kind: fixup_aarch64_movw
// CHECK:   movk w3, #:abs_g0_nc:sym // encoding: [0bAAA00011,A,0b100AAAAA,0x72]
// CHECK-NEXT: // fixup A - offset: 0, value: :abs_g0_nc:sym, kind: fixup_aarch64_movw
// CHECK:   movz x13, #:abs_g0_s:sym // encoding: [0bAAA01101,A,0b100AAAAA,0xd2]
// CHECK-NEXT: // fixup A - offset: 0, value: :abs_g0_s:sym, kind: fixup_aarch64_movw
// CHECK:   movn x17, #:abs_g0_s:sym // encoding: [0bAAA10001,A,0b100AAAAA,0x92]
// CHECK-NEXT: // fixup A - offset: 0, value: :abs_g0_s:sym, kind: fixup_aarch64_movw

// CHECK-OBJ: 0 R_AARCH64_P32_MOVW_UABS_G0 sym
// CHECK-OBJ: 4 R_AARCH64_P32_MOVW_UABS_G0_NC sym
// CHECK-OBJ: 8 R_AARCH64_P32_MOVW_SABS_G0 sym
// CHECK-OBJ: c R_AARCH64_P32_MOVW_SABS_G0 sym

        movz x4, #:abs_g1:sym
// CHECK:   movz x4, #:abs_g1:sym    // encoding: [0bAAA00100,A,0b101AAAAA,0xd2]
// CHECK-NEXT: // fixup A - offset: 0, value: :abs_g1:sym, kind: fixup_aarch64_movw

// CHECK-OBJ: 10 R_AARCH64_P32_MOVW_UABS_G1 sym
