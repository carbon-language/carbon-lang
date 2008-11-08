; RUN: llvm-as < %s | llc -march=ppc32  | grep or | count 3
; This should produce one 'or' or 'cror' instruction per function.

; XFAIL: *

; RUN: llvm-as < %s | llc -march=ppc32  | grep mfcr | count 3
; PR2964

define i32 @test(double %x, double %y) nounwind  {
entry:
	%tmp3 = fcmp ole double %x, %y		; <i1> [#uses=1]
	%tmp345 = zext i1 %tmp3 to i32		; <i32> [#uses=1]
	ret i32 %tmp345
}

define i32 @test2(double %x, double %y) nounwind  {
entry:
	%tmp3 = fcmp one double %x, %y		; <i1> [#uses=1]
	%tmp345 = zext i1 %tmp3 to i32		; <i32> [#uses=1]
	ret i32 %tmp345
}

define i32 @test3(double %x, double %y) nounwind  {
entry:
	%tmp3 = fcmp ugt double %x, %y		; <i1> [#uses=1]
	%tmp34 = zext i1 %tmp3 to i32		; <i32> [#uses=1]
	ret i32 %tmp34
}

