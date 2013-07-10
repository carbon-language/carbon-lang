; RUN: not llc < %s 2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s
; XFAIL: hexagon
; REQUIRES: shell

@0 = global i8 insertvalue( { i8 } select (i1 ptrtoint (i32* @1 to i1), { i8 } { i8 1 }, { i8 } { i8 2 }), i8 0, 0)
@1 = external global i32

; CHECK-ERRORS: Unsupported expression in static initializer: insertvalue

