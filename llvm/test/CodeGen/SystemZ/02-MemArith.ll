; RUN: llc < %s -march=systemz | FileCheck %s

define i32 @foo1(i32 %a, i32 *%b, i64 %idx) signext {
; CHECK: foo1:
; CHECK:  a %r2, 4(%r1,%r3)
entry:
    %idx2 = add i64 %idx, 1         ; <i64> [#uses=1]
    %ptr = getelementptr i32* %b, i64 %idx2          ; <i32*> [#uses=1]
    %c = load i32* %ptr
    %d = add i32 %a, %c
    ret i32 %d
}

define i32 @foo2(i32 %a, i32 *%b, i64 %idx) signext {
; CHECK: foo2:
; CHECK:  ay %r2, -4(%r1,%r3)
entry:
    %idx2 = add i64 %idx, -1         ; <i64> [#uses=1]
    %ptr = getelementptr i32* %b, i64 %idx2          ; <i32*> [#uses=1]
    %c = load i32* %ptr
    %d = add i32 %a, %c
    ret i32 %d
}

define i64 @foo3(i64 %a, i64 *%b, i64 %idx) signext {
; CHECK: foo3:
; CHECK:  ag %r2, 8(%r1,%r3)
entry:
    %idx2 = add i64 %idx, 1         ; <i64> [#uses=1]
    %ptr = getelementptr i64* %b, i64 %idx2          ; <i64*> [#uses=1]
    %c = load i64* %ptr
    %d = add i64 %a, %c
    ret i64 %d
}

define i32 @foo4(i32 %a, i32 *%b, i64 %idx) signext {
; CHECK: foo4:
; CHECK:  n %r2, 4(%r1,%r3)
entry:
    %idx2 = add i64 %idx, 1         ; <i64> [#uses=1]
    %ptr = getelementptr i32* %b, i64 %idx2          ; <i32*> [#uses=1]
    %c = load i32* %ptr
    %d = and i32 %a, %c
    ret i32 %d
}

define i32 @foo5(i32 %a, i32 *%b, i64 %idx) signext {
; CHECK: foo5:
; CHECK:  ny %r2, -4(%r1,%r3)
entry:
    %idx2 = add i64 %idx, -1         ; <i64> [#uses=1]
    %ptr = getelementptr i32* %b, i64 %idx2          ; <i32*> [#uses=1]
    %c = load i32* %ptr
    %d = and i32 %a, %c
    ret i32 %d
}

define i64 @foo6(i64 %a, i64 *%b, i64 %idx) signext {
; CHECK: foo6:
; CHECK:  ng %r2, 8(%r1,%r3)
entry:
    %idx2 = add i64 %idx, 1         ; <i64> [#uses=1]
    %ptr = getelementptr i64* %b, i64 %idx2          ; <i64*> [#uses=1]
    %c = load i64* %ptr
    %d = and i64 %a, %c
    ret i64 %d
}

define i32 @foo7(i32 %a, i32 *%b, i64 %idx) signext {
; CHECK: foo7:
; CHECK:  o %r2, 4(%r1,%r3)
entry:
    %idx2 = add i64 %idx, 1         ; <i64> [#uses=1]
    %ptr = getelementptr i32* %b, i64 %idx2          ; <i32*> [#uses=1]
    %c = load i32* %ptr
    %d = or i32 %a, %c
    ret i32 %d
}

define i32 @foo8(i32 %a, i32 *%b, i64 %idx) signext {
; CHECK: foo8:
; CHECK:  oy %r2, -4(%r1,%r3)
entry:
    %idx2 = add i64 %idx, -1         ; <i64> [#uses=1]
    %ptr = getelementptr i32* %b, i64 %idx2          ; <i32*> [#uses=1]
    %c = load i32* %ptr
    %d = or i32 %a, %c
    ret i32 %d
}

define i64 @foo9(i64 %a, i64 *%b, i64 %idx) signext {
; CHECK: foo9:
; CHECK:  og %r2, 8(%r1,%r3)
entry:
    %idx2 = add i64 %idx, 1         ; <i64> [#uses=1]
    %ptr = getelementptr i64* %b, i64 %idx2          ; <i64*> [#uses=1]
    %c = load i64* %ptr
    %d = or i64 %a, %c
    ret i64 %d
}

define i32 @foo10(i32 %a, i32 *%b, i64 %idx) signext {
; CHECK: foo10:
; CHECK:  x %r2, 4(%r1,%r3)
entry:
    %idx2 = add i64 %idx, 1         ; <i64> [#uses=1]
    %ptr = getelementptr i32* %b, i64 %idx2          ; <i32*> [#uses=1]
    %c = load i32* %ptr
    %d = xor i32 %a, %c
    ret i32 %d
}

define i32 @foo11(i32 %a, i32 *%b, i64 %idx) signext {
; CHECK: foo11:
; CHECK:  xy %r2, -4(%r1,%r3)
entry:
    %idx2 = add i64 %idx, -1         ; <i64> [#uses=1]
    %ptr = getelementptr i32* %b, i64 %idx2          ; <i32*> [#uses=1]
    %c = load i32* %ptr
    %d = xor i32 %a, %c
    ret i32 %d
}

define i64 @foo12(i64 %a, i64 *%b, i64 %idx) signext {
; CHECK: foo12:
; CHECK:  xg %r2, 8(%r1,%r3)
entry:
    %idx2 = add i64 %idx, 1         ; <i64> [#uses=1]
    %ptr = getelementptr i64* %b, i64 %idx2          ; <i64*> [#uses=1]
    %c = load i64* %ptr
    %d = xor i64 %a, %c
    ret i64 %d
}
