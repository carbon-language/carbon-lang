; RUN: llvm-as < %s | llc -march=arm

define csretcc void %__divsc3({ float, float }* %agg.result, float %a, float %b, float %c, float %d) {
entry:
	br i1 false, label %bb, label %cond_next375

bb:		; preds = %entry
	%tmp81 = tail call float %copysignf( float 0x7FF0000000000000, float %c )		; <float> [#uses=1]
	%tmp87 = mul float %tmp81, %b		; <float> [#uses=1]
	br label %cond_next375

cond_next375:		; preds = %bb, %entry
	%y.1 = phi float [ %tmp87, %bb ], [ 0.000000e+00, %entry ]		; <float> [#uses=0]
	ret void
}

declare float %fabsf(float)

declare i1 %llvm.isunordered.f32(float, float)

declare float %copysignf(float, float)
