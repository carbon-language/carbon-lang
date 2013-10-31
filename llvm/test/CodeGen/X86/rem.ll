; RUN: llc < %s -march=x86 | FileCheck %s

; CHECK-LABEL: test1:
; CHECK-NOT: div
define i32 @test1(i32 %X) {
        %tmp1 = srem i32 %X, 255                ; <i32> [#uses=1]
        ret i32 %tmp1
}

; CHECK-LABEL: test2:
; CHECK-NOT: div
define i32 @test2(i32 %X) {
        %tmp1 = srem i32 %X, 256                ; <i32> [#uses=1]
        ret i32 %tmp1
}

; CHECK-LABEL: test3:
; CHECK-NOT: div
define i32 @test3(i32 %X) {
        %tmp1 = urem i32 %X, 255                ; <i32> [#uses=1]
        ret i32 %tmp1
}

; CHECK-LABEL: test4:
; CHECK-NOT: div
define i32 @test4(i32 %X) {
        %tmp1 = urem i32 %X, 256                ; <i32> [#uses=1]
        ret i32 %tmp1
}

; CHECK-LABEL: test5:
; CHECK-NOT: cltd
define i32 @test5(i32 %X) nounwind readnone {
entry:
	%0 = srem i32 41, %X
	ret i32 %0
}
