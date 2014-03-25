; RUN: llc < %s -mcpu=pwr7 -mattr=+vsx | FileCheck %s

; Also run with -schedule-ppc-vsx-fma-mutation-early as a stress test for the
; live-interval-updating logic.
; RUN: llc < %s -mcpu=pwr7 -mattr=+vsx -schedule-ppc-vsx-fma-mutation-early
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @test1(double %a, double %b, double %c, double %e, double* nocapture %d) #0 {
entry:
  %0 = tail call double @llvm.fma.f64(double %b, double %c, double %a)
  store double %0, double* %d, align 8
  %1 = tail call double @llvm.fma.f64(double %b, double %e, double %a)
  %arrayidx1 = getelementptr inbounds double* %d, i64 1
  store double %1, double* %arrayidx1, align 8
  ret void

; CHECK-LABEL: @test1
; CHECK-DAG: li [[C1:[0-9]+]], 8
; CHECK-DAG: xsmaddmdp 3, 2, 1
; CHECK-DAG: xsmaddadp 1, 2, 4
; CHECK-DAG: stxsdx 3, 0, 7
; CHECK-DAG: stxsdx 1, 7, [[C1]]
; CHECK: blr
}

define void @test2(double %a, double %b, double %c, double %e, double %f, double* nocapture %d) #0 {
entry:
  %0 = tail call double @llvm.fma.f64(double %b, double %c, double %a)
  store double %0, double* %d, align 8
  %1 = tail call double @llvm.fma.f64(double %b, double %e, double %a)
  %arrayidx1 = getelementptr inbounds double* %d, i64 1
  store double %1, double* %arrayidx1, align 8
  %2 = tail call double @llvm.fma.f64(double %b, double %f, double %a)
  %arrayidx2 = getelementptr inbounds double* %d, i64 2
  store double %2, double* %arrayidx2, align 8
  ret void

; CHECK-LABEL: @test2
; CHECK-DAG: li [[C1:[0-9]+]], 8
; CHECK-DAG: li [[C2:[0-9]+]], 16
; CHECK-DAG: xsmaddmdp 3, 2, 1
; CHECK-DAG: xsmaddmdp 4, 2, 1
; CHECK-DAG: xsmaddadp 1, 2, 5
; CHECK-DAG: stxsdx 3, 0, 8
; CHECK-DAG: stxsdx 4, 8, [[C1]]
; CHECK-DAG: stxsdx 1, 8, [[C2]]
; CHECK: blr
}

define void @test3(double %a, double %b, double %c, double %e, double %f, double* nocapture %d) #0 {
entry:
  %0 = tail call double @llvm.fma.f64(double %b, double %c, double %a)
  store double %0, double* %d, align 8
  %1 = tail call double @llvm.fma.f64(double %b, double %e, double %a)
  %2 = tail call double @llvm.fma.f64(double %b, double %c, double %1)
  %arrayidx1 = getelementptr inbounds double* %d, i64 3
  store double %2, double* %arrayidx1, align 8
  %3 = tail call double @llvm.fma.f64(double %b, double %f, double %a)
  %arrayidx2 = getelementptr inbounds double* %d, i64 2
  store double %3, double* %arrayidx2, align 8
  %arrayidx3 = getelementptr inbounds double* %d, i64 1
  store double %1, double* %arrayidx3, align 8
  ret void

; CHECK-LABEL: @test3
; CHECK-DAG: xxlor [[F1:[0-9]+]], 1, 1
; CHECK-DAG: li [[C1:[0-9]+]], 24
; CHECK-DAG: li [[C2:[0-9]+]], 16
; CHECK-DAG: li [[C3:[0-9]+]], 8
; CHECK-DAG: xsmaddmdp 4, 2, 1
; CHECK-DAG: xsmaddadp 1, 2, 5

; Note: We could convert this next FMA to M-type as well, but it would require
; re-ordering the instructions.
; CHECK-DAG: xsmaddadp [[F1]], 2, 3

; CHECK-DAG: xsmaddmdp 2, 3, 4
; CHECK-DAG: stxsdx [[F1]], 0, 8
; CHECK-DAG: stxsdx 2, 8, [[C1]]
; CHECK-DAG: stxsdx 1, 8, [[C2]]
; CHECK-DAG: stxsdx 4, 8, [[C3]]
; CHECK-DAG: blr
}

define void @test4(double %a, double %b, double %c, double %e, double %f, double* nocapture %d) #0 {
entry:
  %0 = tail call double @llvm.fma.f64(double %b, double %c, double %a)
  store double %0, double* %d, align 8
  %1 = tail call double @llvm.fma.f64(double %b, double %e, double %a)
  %arrayidx1 = getelementptr inbounds double* %d, i64 1
  store double %1, double* %arrayidx1, align 8
  %2 = tail call double @llvm.fma.f64(double %b, double %c, double %1)
  %arrayidx3 = getelementptr inbounds double* %d, i64 3
  store double %2, double* %arrayidx3, align 8
  %3 = tail call double @llvm.fma.f64(double %b, double %f, double %a)
  %arrayidx4 = getelementptr inbounds double* %d, i64 2
  store double %3, double* %arrayidx4, align 8
  ret void

; CHECK-LABEL: @test4
; CHECK-DAG: xxlor [[F1:[0-9]+]], 1, 1
; CHECK-DAG: li [[C1:[0-9]+]], 8
; CHECK-DAG: li [[C2:[0-9]+]], 16
; CHECK-DAG: xsmaddmdp 4, 2, 1

; Note: We could convert this next FMA to M-type as well, but it would require
; re-ordering the instructions.
; CHECK-DAG: xsmaddadp 1, 2, 5

; CHECK-DAG: xsmaddadp [[F1]], 2, 3
; CHECK-DAG: stxsdx [[F1]], 0, 8
; CHECK-DAG: stxsdx 4, 8, [[C1]]
; CHECK-DAG: li [[C3:[0-9]+]], 24
; CHECK-DAG: xsmaddadp 4, 2, 3
; CHECK-DAG: stxsdx 4, 8, [[C3]]
; CHECK-DAG: stxsdx 1, 8, [[C2]]
; CHECK: blr
}

declare double @llvm.fma.f64(double, double, double) #0

attributes #0 = { nounwind readnone }

