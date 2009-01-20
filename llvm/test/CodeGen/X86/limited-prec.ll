; RUN: llvm-as < %s | llc -limit-float-precision=6 -march=x86 | \
; RUN:    not grep exp | not grep log | not grep pow
; RUN: llvm-as < %s | llc -limit-float-precision=12 -march=x86 | \
; RUN:    not grep exp | not grep log | not grep pow
; RUN: llvm-as < %s | llc -limit-float-precision=18 -march=x86 | \
; RUN:    not grep exp | not grep log | not grep pow
target triple = "i386-apple-darwin9.5"

define float @f1(float %x) nounwind noinline {
entry:
	%x_addr = alloca float		; <float*> [#uses=2]
	%retval = alloca float		; <float*> [#uses=2]
	%0 = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float %x, float* %x_addr
	%1 = load float* %x_addr, align 4		; <float> [#uses=1]
	%2 = call float @llvm.exp.f32(float %1)		; <float> [#uses=1]
	store float %2, float* %0, align 4
	%3 = load float* %0, align 4		; <float> [#uses=1]
	store float %3, float* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load float* %retval		; <float> [#uses=1]
	ret float %retval1
}

declare float @llvm.exp.f32(float) nounwind readonly

define float @f2(float %x) nounwind noinline {
entry:
	%x_addr = alloca float		; <float*> [#uses=2]
	%retval = alloca float		; <float*> [#uses=2]
	%0 = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float %x, float* %x_addr
	%1 = load float* %x_addr, align 4		; <float> [#uses=1]
	%2 = call float @llvm.exp2.f32(float %1)		; <float> [#uses=1]
	store float %2, float* %0, align 4
	%3 = load float* %0, align 4		; <float> [#uses=1]
	store float %3, float* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load float* %retval		; <float> [#uses=1]
	ret float %retval1
}

declare float @llvm.exp2.f32(float) nounwind readonly

define float @f3(float %x) nounwind noinline {
entry:
	%x_addr = alloca float		; <float*> [#uses=2]
	%retval = alloca float		; <float*> [#uses=2]
	%0 = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float %x, float* %x_addr
	%1 = load float* %x_addr, align 4		; <float> [#uses=1]
	%2 = call float @llvm.pow.f32(float 1.000000e+01, float %1)		; <float> [#uses=1]
	store float %2, float* %0, align 4
	%3 = load float* %0, align 4		; <float> [#uses=1]
	store float %3, float* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load float* %retval		; <float> [#uses=1]
	ret float %retval1
}

declare float @llvm.pow.f32(float, float) nounwind readonly

define float @f4(float %x) nounwind noinline {
entry:
	%x_addr = alloca float		; <float*> [#uses=2]
	%retval = alloca float		; <float*> [#uses=2]
	%0 = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float %x, float* %x_addr
	%1 = load float* %x_addr, align 4		; <float> [#uses=1]
	%2 = call float @llvm.log.f32(float %1)		; <float> [#uses=1]
	store float %2, float* %0, align 4
	%3 = load float* %0, align 4		; <float> [#uses=1]
	store float %3, float* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load float* %retval		; <float> [#uses=1]
	ret float %retval1
}

declare float @llvm.log.f32(float) nounwind readonly

define float @f5(float %x) nounwind noinline {
entry:
	%x_addr = alloca float		; <float*> [#uses=2]
	%retval = alloca float		; <float*> [#uses=2]
	%0 = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float %x, float* %x_addr
	%1 = load float* %x_addr, align 4		; <float> [#uses=1]
	%2 = call float @llvm.log2.f32(float %1)		; <float> [#uses=1]
	store float %2, float* %0, align 4
	%3 = load float* %0, align 4		; <float> [#uses=1]
	store float %3, float* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load float* %retval		; <float> [#uses=1]
	ret float %retval1
}

declare float @llvm.log2.f32(float) nounwind readonly

define float @f6(float %x) nounwind noinline {
entry:
	%x_addr = alloca float		; <float*> [#uses=2]
	%retval = alloca float		; <float*> [#uses=2]
	%0 = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store float %x, float* %x_addr
	%1 = load float* %x_addr, align 4		; <float> [#uses=1]
	%2 = call float @llvm.log10.f32(float %1)		; <float> [#uses=1]
	store float %2, float* %0, align 4
	%3 = load float* %0, align 4		; <float> [#uses=1]
	store float %3, float* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load float* %retval		; <float> [#uses=1]
	ret float %retval1
}

declare float @llvm.log10.f32(float) nounwind readonly
