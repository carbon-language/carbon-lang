; RUN: llc  %s -mtriple=arm-linux-gnueabi -filetype=obj -o - | \
; RUN:    elf-dump --dump-section-data | FileCheck %s
; This tests that the extpected ARM attributes are emitted.
;
; CHECK:        .ARM.attributes
; CHECK-NEXT:         0x70000003
; CHECK-NEXT:         0x00000000
; CHECK-NEXT:         0x00000000
; CHECK-NEXT:         0x0000003c
; CHECK-NEXT:         0x00000022
; CHECK-NEXT:         0x00000000
; CHECK-NEXT:         0x00000000
; CHECK-NEXT:         0x00000001
; CHECK-NEXT:         0x00000000
; CHECK-NEXT:         '41210000 00616561 62690001 17000000 06020801 09011401 15011703 18011901 2c01'

define i32 @f(i64 %z) {
       ret i32 0
}
