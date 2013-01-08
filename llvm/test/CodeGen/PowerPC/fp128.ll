; RUN: llc -mtriple=powerpc64-none-linux < %s | FileCheck --check-prefix=BIGENDIAN %s

@var = global fp128 0xL00000000000000008000000000000000

; CHECK-BIGENDIAN: var:
; CHECK-BIGENDIAN-NEXT: .quad   -9223372036854775808    # fp128 -0
; CHECK-BIGENDIAN-NEXT: .quad   0

