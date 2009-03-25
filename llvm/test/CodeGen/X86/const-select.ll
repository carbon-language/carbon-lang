
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

; RUN: llvm-as < %s | llc | grep {LCPI1_0(,%eax,4)}
define float @f(i32 %x) nounwind readnone {
entry:
	%0 = icmp eq i32 %x, 0		; <i1> [#uses=1]
	%iftmp.0.0 = select i1 %0, float 4.200000e+01, float 2.300000e+01		; <float> [#uses=1]
	ret float %iftmp.0.0
}

; RUN: llvm-as < %s | llc | grep {movsbl.*(%e.x,%e.x,4), %eax}
define signext i8 @test(i8* nocapture %P, double %F) nounwind readonly {
entry:
	%0 = fcmp olt double %F, 4.200000e+01		; <i1> [#uses=1]
	%iftmp.0.0 = select i1 %0, i32 4, i32 0		; <i32> [#uses=1]
	%1 = getelementptr i8* %P, i32 %iftmp.0.0		; <i8*> [#uses=1]
	%2 = load i8* %1, align 1		; <i8> [#uses=1]
	ret i8 %2
}

