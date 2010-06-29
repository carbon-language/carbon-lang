; RUN: llc < %s -O0 -mcpu=cortex-a8 | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; This tests the fast register allocator's handling of partial redefines:
;
;	%reg1026<def> = VMOVv16i8 0, pred:14, pred:%reg0
;	%reg1028:dsub_1<def> = EXTRACT_SUBREG %reg1026<kill>, 1
;
; %reg1026 gets allocated %Q0, and if %reg1028 is reloaded for the partial redef,
; it cannot also get %Q0.

; CHECK: vmov.i8 q0, #0x0
; CHECK-NOT: vld1.64 {d0,d1}
; CHECK: vmov.f64 d3, d0

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
 %0 = shufflevector <2 x i64> undef, <2 x i64> zeroinitializer, <2 x i32> <i32 1, i32 2> ; <<2 x i64>> [#uses=1]
 store <2 x i64> %0, <2 x i64>* undef, align 16
 ret i32 undef
}
