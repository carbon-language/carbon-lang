; RUN: llc < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; Currently there is no custom lowering for vector shuffles operating on types
; bigger than NEON. However, having no support opens us up to a code generator
; hang when expanding BUILD_VECTOR. Here we just validate the promblematic case
; successfully exits code generation.
define void @hang_when_merging_stores_after_legalisation(<8 x i32>* %a, <2 x i32> %b) #0 {
; CHECK-LABEL: hang_when_merging_stores_after_legalisation:
  %splat = shufflevector <2 x i32> %b, <2 x i32> undef, <8 x i32> zeroinitializer
  %interleaved.vec = shufflevector <8 x i32> %splat, <8 x i32> undef, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x i32> %interleaved.vec, <8 x i32>* %a, align 4
  ret void
}

; Ensure we don't crash when trying to lower a shuffle via and extract
define void @crash_when_lowering_extract_shuffle(<32 x i32>* %dst, i1 %cond) #0 {
; CHECK-LABEL: crash_when_lowering_extract_shuffle:
; CHECK:       ld1w { z3.s }, p0/z, [x0]
; CHECK:       st1w { z3.s }, p0, [x0]
  %broadcast.splat = shufflevector <32 x i1> zeroinitializer, <32 x i1> zeroinitializer, <32 x i32> zeroinitializer
  br i1 %cond, label %exit, label %vector.body

vector.body:
  %1 = load <32 x i32>, <32 x i32>* %dst, align 16
  %predphi = select <32 x i1> %broadcast.splat, <32 x i32> zeroinitializer, <32 x i32> %1
  store <32 x i32> %predphi, <32 x i32>* %dst, align 16
  br label %exit

exit:
  ret void
}

attributes #0 = { vscale_range(2,2) "target-features"="+sve" }
