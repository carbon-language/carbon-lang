; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-bgq-linux"

define void @_Z15quad_copy_1024nPcS_m() nounwind {
; CHECK: @_Z15quad_copy_1024nPcS_m

entry:
  br i1 undef, label %short_msg, label %if.end

if.end:                                           ; preds = %entry
  %0 = tail call double* asm sideeffect "qvstfdux $2,$0,$1", "=b,{r7},{f11},0,~{memory}"(i32 64, double undef, double* undef) nounwind, !srcloc !0
  unreachable

; CHECK: qvstfdux 11,{{[0-9]+}},7

short_msg:                                        ; preds = %entry
  ret void
}

!0 = metadata !{i32 -2147422199}                  
