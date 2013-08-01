; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s -check-prefix=NO-REALIGN
; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s -check-prefix=REALIGN

; rdar://12713765
; When realign-stack is set to false, make sure we are not creating stack
; objects that are assumed to be 64-byte aligned.
@T3_retval = common global <16 x float> zeroinitializer, align 16

define void @test1(<16 x float>* noalias sret %agg.result) nounwind ssp "no-realign-stack" {
entry:
; NO-REALIGN: test1
; NO-REALIGN: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #48
; NO-REALIGN: vst1.64
; NO-REALIGN: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #32
; NO-REALIGN: vst1.64
; NO-REALIGN: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #16
; NO-REALIGN: vst1.64
; NO-REALIGN: vst1.64
; NO-REALIGN: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #48
; NO-REALIGN: vst1.64
; NO-REALIGN: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #32
; NO-REALIGN: vst1.64
; NO-REALIGN: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #16
; NO-REALIGN: vst1.64
; NO-REALIGN: vst1.64
 %retval = alloca <16 x float>, align 16
 %0 = load <16 x float>* @T3_retval, align 16
 store <16 x float> %0, <16 x float>* %retval
 %1 = load <16 x float>* %retval
 store <16 x float> %1, <16 x float>* %agg.result, align 16
 ret void
}

define void @test2(<16 x float>* noalias sret %agg.result) nounwind ssp {
entry:
; REALIGN: test2
; REALIGN: bic sp, sp, #63
; REALIGN: orr [[R2:r[0-9]+]], [[R1:r[0-9]+]], #48
; REALIGN: vst1.64
; REALIGN: orr [[R2:r[0-9]+]], [[R1:r[0-9]+]], #32
; REALIGN: vst1.64
; REALIGN: orr [[R2:r[0-9]+]], [[R1:r[0-9]+]], #16
; REALIGN: vst1.64
; REALIGN: vst1.64
; REALIGN: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #48
; REALIGN: vst1.64
; REALIGN: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #32
; REALIGN: vst1.64
; REALIGN: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #16
; REALIGN: vst1.64
; REALIGN: vst1.64
 %retval = alloca <16 x float>, align 16
 %0 = load <16 x float>* @T3_retval, align 16
 store <16 x float> %0, <16 x float>* %retval
 %1 = load <16 x float>* %retval
 store <16 x float> %1, <16 x float>* %agg.result, align 16
 ret void
}
