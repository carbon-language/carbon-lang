; RUN: llc < %s -march=sparc | FileCheck %s

define i32 @test0(i32 %X) {
	%tmp.1 = add i32 %X, 1
	ret i32 %tmp.1
; CHECK-LABEL: test0:
; CHECK: add %o0, 1, %o0
}


;; xnor tests.
define i32 @test1(i32 %X, i32 %Y) {
        %A = xor i32 %X, %Y
        %B = xor i32 %A, -1
        ret i32 %B
; CHECK-LABEL: test1:
; CHECK: xnor %o0, %o1, %o0
}

define i32 @test2(i32 %X, i32 %Y) {
        %A = xor i32 %X, -1
        %B = xor i32 %A, %Y
        ret i32 %B
; CHECK-LABEL: test2:
; CHECK: xnor %o0, %o1, %o0
}

; CHECK: store_zero:
; CHECK: st   %g0, [%o0]
; CHECK: st   %g0, [%o1+4]
define i32 @store_zero(i32* %a, i32* %b) {
entry:
  store i32 0, i32* %a, align 4
  %0 = getelementptr inbounds i32* %b, i32 1
  store i32 0, i32* %0, align 4
  ret i32 0
}

