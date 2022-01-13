; RUN: llc < %s 
; rdar://5763967

define void @test() {
entry:
	%tmp98 = load float, float* null, align 4		; <float> [#uses=1]
	%tmp106 = load float, float* null, align 4		; <float> [#uses=1]
	%tmp113 = fadd float %tmp98, %tmp106		; <float> [#uses=1]
	%tmp119 = fsub float %tmp113, 0.000000e+00		; <float> [#uses=1]
	call void (i32, ...) @foo( i32 0, float 0.000000e+00, float %tmp119 ) nounwind 
	ret void
}

declare void @foo(i32, ...)
