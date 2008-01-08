; RUN: llvm-as < %s | llc -march=ppc32 | grep or | count 3
; This should produce one 'or' or 'cror' instruction per function.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

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

