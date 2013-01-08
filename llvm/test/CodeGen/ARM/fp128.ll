; RUN: llc -mtriple=arm-none-linux < %s | FileCheck --check-prefix=LITTLEENDIAN %s

@var = global fp128 0xL00000000000000008000000000000000

; CHECK-LITTLEENDIAN: var:
; CHECK-LITTLEENDIAN-NEXT: .long   0                       @ fp128 -0
; CHECK-LITTLEENDIAN-NEXT: .long   0
; CHECK-LITTLEENDIAN-NEXT: .long   0
; CHECK-LITTLEENDIAN-NEXT: .long   2147483648

