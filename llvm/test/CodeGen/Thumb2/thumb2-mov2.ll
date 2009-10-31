; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @t2MOVTi16_ok_1(i32 %a) {
; CHECK: t2MOVTi16_ok_1:
; CHECK: movt r0, #1234
    %1 = and i32 %a, 65535
    %2 = shl i32 1234, 16
    %3 = or  i32 %1, %2

    ret i32 %3
}

define i32 @t2MOVTi16_test_1(i32 %a) {
; CHECK: t2MOVTi16_test_1:
; CHECK: movt r0, #1234
    %1 = shl i32  255,   8
    %2 = shl i32 1234,   8
    %3 = or  i32   %1, 255  ; This gives us 0xFFFF in %3
    %4 = shl i32   %2,   8  ; This gives us (1234 << 16) in %4
    %5 = and i32   %a,  %3
    %6 = or  i32   %4,  %5

    ret i32 %6
}

define i32 @t2MOVTi16_test_2(i32 %a) {
; CHECK: t2MOVTi16_test_2:
; CHECK: movt r0, #1234
    %1 = shl i32  255,   8
    %2 = shl i32 1234,   8
    %3 = or  i32   %1, 255  ; This gives us 0xFFFF in %3
    %4 = shl i32   %2,   6
    %5 = and i32   %a,  %3
    %6 = shl i32   %4,   2  ; This gives us (1234 << 16) in %6
    %7 = or  i32   %5,  %6

    ret i32 %7
}

define i32 @t2MOVTi16_test_3(i32 %a) {
; CHECK: t2MOVTi16_test_3:
; CHECK: movt r0, #1234
    %1 = shl i32  255,   8
    %2 = shl i32 1234,   8
    %3 = or  i32   %1, 255  ; This gives us 0xFFFF in %3
    %4 = shl i32   %2,   6
    %5 = and i32   %a,  %3
    %6 = shl i32   %4,   2  ; This gives us (1234 << 16) in %6
    %7 = lshr i32  %6,   6
    %8 = shl i32   %7,   6
    %9 = or  i32   %5,  %8

    ret i32 %9
}

define i32 @t2MOVTi16_test_nomatch_1(i32 %a) {
; CHECK: t2MOVTi16_test_nomatch_1:
; CHECK:      #8388608
; CHECK:      movw r1, #65535
; CHECK-NEXT: movt r1, #154
; CHECK:      #1720320
    %1 = shl i32  255,   8
    %2 = shl i32 1234,   8
    %3 = or  i32   %1, 255  ; This gives us 0xFFFF in %3
    %4 = shl i32   %2,   6
    %5 = and i32   %a,  %3
    %6 = shl i32   %4,   2  ; This gives us (1234 << 16) in %6
    %7 = lshr i32  %6,   3
    %8 = or  i32   %5,  %7
    ret i32 %8
}


