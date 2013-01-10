; RUN: llc < %s -mtriple=armv7-apple-ios -O0 -realign-stack=0 2>&1 | FileCheck %s

; rdar://12713765
@T3_retval = common global <16 x float> zeroinitializer, align 16

; If alignment for alloc is smaller than or equal to stack alignment, but the 
; preferred type alignment is bigger, the alignment will be clamped.
; If alignment for alloca is bigger than stack alignment, the compiler
; will emit an error.
define void @test(<16 x float>* noalias sret %agg.result) nounwind ssp {
entry:
; CHECK: Requested Minimal Alignment exceeds the Stack Alignment!
 %retval = alloca <16 x float>, align 16
 %0 = load <16 x float>* @T3_retval, align 16
 store <16 x float> %0, <16 x float>* %retval
 %1 = load <16 x float>* %retval
 store <16 x float> %1, <16 x float>* %agg.result, align 16
 ret void
}
