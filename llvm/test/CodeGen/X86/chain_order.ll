; RUN: llc < %s -mcpu=corei7-avx -mtriple=x86_64-linux | FileCheck %s

;CHECK-LABEL: cftx020:
;CHECK: vmovsd  (%rdi), %xmm{{.*}}
;CHECK: vmovsd  16(%rdi), %xmm{{.*}}
;CHECK: vmovhpd  8(%rdi), %xmm{{.*}}
;CHECK: vmovsd  24(%rdi), %xmm{{.*}}
;CHECK: vmovupd %xmm{{.*}}, (%rdi)
;CHECK: vmovupd %xmm{{.*}}, 16(%rdi)
;CHECK: ret

; A test from pifft (after SLP-vectorization) that fails when we drop the chain on newly merged loads.
define void @cftx020(double* nocapture %a) {
entry:
  %0 = load double* %a, align 8
  %arrayidx1 = getelementptr inbounds double* %a, i64 2
  %1 = load double* %arrayidx1, align 8
  %arrayidx2 = getelementptr inbounds double* %a, i64 1
  %2 = load double* %arrayidx2, align 8
  %arrayidx3 = getelementptr inbounds double* %a, i64 3
  %3 = load double* %arrayidx3, align 8
  %4 = insertelement <2 x double> undef, double %0, i32 0
  %5 = insertelement <2 x double> %4, double %3, i32 1
  %6 = insertelement <2 x double> undef, double %1, i32 0
  %7 = insertelement <2 x double> %6, double %2, i32 1
  %8 = fadd <2 x double> %5, %7
  %9 = bitcast double* %a to <2 x double>*
  store <2 x double> %8, <2 x double>* %9, align 8
  %10 = insertelement <2 x double> undef, double %0, i32 0
  %11 = insertelement <2 x double> %10, double %2, i32 1
  %12 = insertelement <2 x double> undef, double %1, i32 0
  %13 = insertelement <2 x double> %12, double %3, i32 1
  %14 = fsub <2 x double> %11, %13
  %15 = bitcast double* %arrayidx1 to <2 x double>*
  store <2 x double> %14, <2 x double>* %15, align 8
  ret void
}

