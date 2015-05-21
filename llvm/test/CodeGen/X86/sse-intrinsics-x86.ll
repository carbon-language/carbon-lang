; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=-avx,+sse | FileCheck %s
; RUN: llc < %s -mtriple=i386-apple-darwin -mcpu=knl | FileCheck %s

define <4 x float> @test_x86_sse_add_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: addss
  %res = call <4 x float> @llvm.x86.sse.add.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.add.ss(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_cmp_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: cmpordps
  %res = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %a0, <4 x float> %a1, i8 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone


define <4 x float> @test_x86_sse_cmp_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: cmpordss
  %res = call <4 x float> @llvm.x86.sse.cmp.ss(<4 x float> %a0, <4 x float> %a1, i8 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.cmp.ss(<4 x float>, <4 x float>, i8) nounwind readnone


define i32 @test_x86_sse_comieq_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: comiss
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.comieq.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comieq.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_comige_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: comiss
  ; CHECK: setae
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.comige.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comige.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_comigt_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: comiss
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.comigt.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comigt.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_comile_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: comiss
  ; CHECK: setbe
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.comile.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comile.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_comilt_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: comiss
  ; CHECK: sbb
  %res = call i32 @llvm.x86.sse.comilt.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comilt.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_comineq_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: comiss
  ; CHECK: setne
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.comineq.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.comineq.ss(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_cvtsi2ss(<4 x float> %a0) {
  ; CHECK: movl
  ; CHECK: cvtsi2ss
  %res = call <4 x float> @llvm.x86.sse.cvtsi2ss(<4 x float> %a0, i32 7) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.cvtsi2ss(<4 x float>, i32) nounwind readnone


define i32 @test_x86_sse_cvtss2si(<4 x float> %a0) {
  ; CHECK: cvtss2si
  %res = call i32 @llvm.x86.sse.cvtss2si(<4 x float> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.cvtss2si(<4 x float>) nounwind readnone


define i32 @test_x86_sse_cvttss2si(<4 x float> %a0) {
  ; CHECK: cvttss2si
  %res = call i32 @llvm.x86.sse.cvttss2si(<4 x float> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.cvttss2si(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_div_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: divss
  %res = call <4 x float> @llvm.x86.sse.div.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.div.ss(<4 x float>, <4 x float>) nounwind readnone


define void @test_x86_sse_ldmxcsr(i8* %a0) {
  ; CHECK: movl
  ; CHECK: ldmxcsr
  call void @llvm.x86.sse.ldmxcsr(i8* %a0)
  ret void
}
declare void @llvm.x86.sse.ldmxcsr(i8*) nounwind



define <4 x float> @test_x86_sse_max_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: maxps
  %res = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_max_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: maxss
  %res = call <4 x float> @llvm.x86.sse.max.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_min_ps(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: minps
  %res = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_min_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: minss
  %res = call <4 x float> @llvm.x86.sse.min.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_movmsk_ps(<4 x float> %a0) {
  ; CHECK: movmskps
  %res = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %a0) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.movmsk.ps(<4 x float>) nounwind readnone



define <4 x float> @test_x86_sse_mul_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: mulss
  %res = call <4 x float> @llvm.x86.sse.mul.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.mul.ss(<4 x float>, <4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_rcp_ps(<4 x float> %a0) {
  ; CHECK: rcpps
  %res = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_rcp_ss(<4 x float> %a0) {
  ; CHECK: rcpss
  %res = call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_rsqrt_ps(<4 x float> %a0) {
  ; CHECK: rsqrtps
  %res = call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_rsqrt_ss(<4 x float> %a0) {
  ; CHECK: rsqrtss
  %res = call <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_sqrt_ps(<4 x float> %a0) {
  ; CHECK: sqrtps
  %res = call <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_sqrt_ss(<4 x float> %a0) {
  ; CHECK: sqrtss
  %res = call <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone


define void @test_x86_sse_stmxcsr(i8* %a0) {
  ; CHECK: movl
  ; CHECK: stmxcsr
  call void @llvm.x86.sse.stmxcsr(i8* %a0)
  ret void
}
declare void @llvm.x86.sse.stmxcsr(i8*) nounwind


define void @test_x86_sse_storeu_ps(i8* %a0, <4 x float> %a1) {
  ; CHECK: movl
  ; CHECK: movups
  call void @llvm.x86.sse.storeu.ps(i8* %a0, <4 x float> %a1)
  ret void
}
declare void @llvm.x86.sse.storeu.ps(i8*, <4 x float>) nounwind


define <4 x float> @test_x86_sse_sub_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: subss
  %res = call <4 x float> @llvm.x86.sse.sub.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.sub.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomieq_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: ucomiss
  ; CHECK: sete
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.ucomieq.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomieq.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomige_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: ucomiss
  ; CHECK: setae
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.ucomige.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomige.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomigt_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: ucomiss
  ; CHECK: seta
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.ucomigt.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomigt.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomile_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: ucomiss
  ; CHECK: setbe
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.ucomile.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomile.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomilt_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: ucomiss
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.sse.ucomilt.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomilt.ss(<4 x float>, <4 x float>) nounwind readnone


define i32 @test_x86_sse_ucomineq_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: ucomiss
  ; CHECK: setne
  ; CHECK: movzbl
  %res = call i32 @llvm.x86.sse.ucomineq.ss(<4 x float> %a0, <4 x float> %a1) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse.ucomineq.ss(<4 x float>, <4 x float>) nounwind readnone
