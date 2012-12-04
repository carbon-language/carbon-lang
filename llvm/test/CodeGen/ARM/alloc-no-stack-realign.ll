; RUN: llc < %s -mtriple=armv7-apple-ios -O0 -realign-stack=0 | FileCheck %s -check-prefix=NO-REALIGN
; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s

; rdar://12713765
; When realign-stack is set to false, make sure we are not creating stack
; objects that are assumed to be 64-byte aligned.
@T3_retval = common global <16 x float> zeroinitializer, align 16

define void @test(<16 x float>* noalias sret %agg.result) nounwind ssp {
entry:
; CHECK: test
; CHECK: bic sp, sp, #63
; CHECK: orr [[R2:r[0-9]+]], [[R1:r[0-9]+]], #48
; CHECK: vst1.64
; CHECK: orr [[R2:r[0-9]+]], [[R1:r[0-9]+]], #32
; CHECK: vst1.64
; CHECK: orr [[R2:r[0-9]+]], [[R1:r[0-9]+]], #16
; CHECK: vst1.64
; CHECK: vst1.64
; CHECK: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #48
; CHECK: vst1.64
; CHECK: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #32
; CHECK: vst1.64
; CHECK: add [[R2:r[0-9]+]], [[R1:r[0-9]+]], #16
; CHECK: vst1.64
; CHECK: vst1.64
; NO-REALIGN: test
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
