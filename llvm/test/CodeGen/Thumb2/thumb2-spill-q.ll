; RUN: llc < %s -mtriple=thumbv7-elf -mattr=+neon | FileCheck %s
; PR4789

%bar = type { float, float, float }
%baz = type { i32, [16 x %bar], [16 x float], [16 x i32], i8 }
%foo = type { <4 x float> }
%quux = type { i32 (...)**, %baz*, i32 }
%quuz = type { %quux, i32, %bar, [128 x i8], [16 x %foo], %foo, %foo, %foo }

declare <4 x float> @llvm.arm.neon.vld1.v4f32(i8*) nounwind readonly

define arm_apcscc void @aaa(%quuz* %this, i8* %block) {
; CHECK: aaa:
; CHECK: bic r4, r4, #15
; CHECK: vst1.64 {{.*}}[{{.*}}, :128]
; CHECK: vld1.64 {{.*}}[{{.*}}, :128]
entry:
  %0 = call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* undef) nounwind ; <<4 x float>> [#uses=1]
  store float 6.300000e+01, float* undef, align 4
  %1 = call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* undef) nounwind ; <<4 x float>> [#uses=1]
  store float 0.000000e+00, float* undef, align 4
  %2 = call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* undef) nounwind ; <<4 x float>> [#uses=1]
  %val173 = load <4 x float>* undef               ; <<4 x float>> [#uses=1]
  br label %bb4

bb4:                                              ; preds = %bb193, %entry
  %besterror.0.2264 = phi <4 x float> [ undef, %entry ], [ %besterror.0.0, %bb193 ] ; <<4 x float>> [#uses=2]
  %part0.0.0261 = phi <4 x float> [ zeroinitializer, %entry ], [ %23, %bb193 ] ; <<4 x float>> [#uses=2]
  %3 = fmul <4 x float> zeroinitializer, %0       ; <<4 x float>> [#uses=2]
  %4 = fadd <4 x float> %3, %part0.0.0261         ; <<4 x float>> [#uses=1]
  %5 = shufflevector <4 x float> %3, <4 x float> undef, <2 x i32> <i32 2, i32 3> ; <<2 x float>> [#uses=1]
  %6 = shufflevector <2 x float> %5, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x float>> [#uses=1]
  %7 = fmul <4 x float> %1, undef                 ; <<4 x float>> [#uses=1]
  %8 = fadd <4 x float> %7, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01> ; <<4 x float>> [#uses=1]
  %9 = fptosi <4 x float> %8 to <4 x i32>         ; <<4 x i32>> [#uses=1]
  %10 = sitofp <4 x i32> %9 to <4 x float>        ; <<4 x float>> [#uses=1]
  %11 = fmul <4 x float> %10, %2                  ; <<4 x float>> [#uses=1]
  %12 = fmul <4 x float> undef, %6                ; <<4 x float>> [#uses=1]
  %13 = fmul <4 x float> %11, %4                  ; <<4 x float>> [#uses=1]
  %14 = fsub <4 x float> %12, %13                 ; <<4 x float>> [#uses=1]
  %15 = fsub <4 x float> %14, undef               ; <<4 x float>> [#uses=1]
  %16 = fmul <4 x float> %15, <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00> ; <<4 x float>> [#uses=1]
  %17 = fadd <4 x float> %16, undef               ; <<4 x float>> [#uses=1]
  %18 = fmul <4 x float> %17, %val173             ; <<4 x float>> [#uses=1]
  %19 = shufflevector <4 x float> %18, <4 x float> undef, <2 x i32> <i32 2, i32 3> ; <<2 x float>> [#uses=1]
  %20 = shufflevector <2 x float> %19, <2 x float> undef, <4 x i32> zeroinitializer ; <<4 x float>> [#uses=1]
  %21 = fadd <4 x float> zeroinitializer, %20     ; <<4 x float>> [#uses=2]
  %22 = fcmp ogt <4 x float> %besterror.0.2264, %21 ; <<4 x i1>> [#uses=0]
  br i1 undef, label %bb193, label %bb186

bb186:                                            ; preds = %bb4
  br label %bb193

bb193:                                            ; preds = %bb186, %bb4
  %besterror.0.0 = phi <4 x float> [ %21, %bb186 ], [ %besterror.0.2264, %bb4 ] ; <<4 x float>> [#uses=1]
  %23 = fadd <4 x float> %part0.0.0261, zeroinitializer ; <<4 x float>> [#uses=1]
  br label %bb4
}
