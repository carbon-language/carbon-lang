; RUN: opt < %s -S -loop-unroll -mcpu=nehalem -x86-use-partial-unrolling=1 | FileCheck %s
; RUN: opt < %s -S -loop-unroll -mcpu=core -x86-use-partial-unrolling=1 | FileCheck -check-prefix=CHECK-NOUNRL %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32* noalias nocapture readnone %ip, double %alpha, double* noalias nocapture %a, double* noalias nocapture readonly %b) #0 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds double* %b, i64 %index
  %1 = bitcast double* %0 to <2 x double>*
  %wide.load = load <2 x double>* %1, align 8
  %.sum9 = or i64 %index, 2
  %2 = getelementptr double* %b, i64 %.sum9
  %3 = bitcast double* %2 to <2 x double>*
  %wide.load8 = load <2 x double>* %3, align 8
  %4 = fadd <2 x double> %wide.load, <double 1.000000e+00, double 1.000000e+00>
  %5 = fadd <2 x double> %wide.load8, <double 1.000000e+00, double 1.000000e+00>
  %6 = getelementptr inbounds double* %a, i64 %index
  %7 = bitcast double* %6 to <2 x double>*
  store <2 x double> %4, <2 x double>* %7, align 8
  %.sum10 = or i64 %index, 2
  %8 = getelementptr double* %a, i64 %.sum10
  %9 = bitcast double* %8 to <2 x double>*
  store <2 x double> %5, <2 x double>* %9, align 8
  %index.next = add i64 %index, 4
  %10 = icmp eq i64 %index.next, 1600
  br i1 %10, label %for.end, label %vector.body

; FIXME: We should probably unroll this loop by a factor of 2, but the cost
; model needs to be fixed to account for instructions likely to be folded
; as part of an addressing mode.
; CHECK-LABEL: @foo
; CHECK-NOUNRL-LABEL: @foo

for.end:                                          ; preds = %vector.body
  ret void
}

define void @bar(i32* noalias nocapture readnone %ip, double %alpha, double* noalias nocapture %a, double* noalias nocapture readonly %b) #0 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %v0 = getelementptr inbounds double* %b, i64 %index
  %v1 = bitcast double* %v0 to <2 x double>*
  %wide.load = load <2 x double>* %v1, align 8
  %v4 = fadd <2 x double> %wide.load, <double 1.000000e+00, double 1.000000e+00>
  %v5 = fmul <2 x double> %v4, <double 8.000000e+00, double 8.000000e+00>
  %v6 = getelementptr inbounds double* %a, i64 %index
  %v7 = bitcast double* %v6 to <2 x double>*
  store <2 x double> %v5, <2 x double>* %v7, align 8
  %index.next = add i64 %index, 2
  %v10 = icmp eq i64 %index.next, 1600
  br i1 %v10, label %for.end, label %vector.body

; FIXME: We should probably unroll this loop by a factor of 2, but the cost
; model needs to first to fixed to account for instructions likely to be folded
; as part of an addressing mode.

; CHECK-LABEL: @bar
; CHECK: fadd
; CHECK-NEXT: fmul
; CHECK: fadd
; CHECK-NEXT: fmul

; CHECK-NOUNRL-LABEL: @bar
; CHECK-NOUNRL: fadd
; CHECK-NOUNRL-NEXT: fmul
; CHECK-NOUNRL-NOT: fadd

for.end:                                          ; preds = %vector.body
  ret void
}

attributes #0 = { nounwind uwtable }

