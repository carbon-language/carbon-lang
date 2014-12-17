; RUN: llc -O3 -disable-peephole -mcpu=corei7-avx -mattr=+avx < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests - we use the 'big vectors' pattern to guarantee spilling to stack.
;
; Many of these tests are primarily to check memory folding with specific instructions. Using a basic
; load/cvt/store pattern to test for this would mean that it wouldn't be the memory folding code thats
; being tested - the load-execute version of the instruction from the tables would be matched instead.

define void @stack_fold_vmulpd(<64 x double>* %a, <64 x double>* %b, <64 x double>* %c) {
  ;CHECK-LABEL: stack_fold_vmulpd
  ;CHECK:       vmulpd {{[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload

  %1 = load <64 x double>* %a
  %2 = load <64 x double>* %b
  %3 = fadd <64 x double> %1, %2
  %4 = fsub <64 x double> %1, %2
  %5 = fmul <64 x double> %3, %4
  store <64 x double> %5, <64 x double>* %c
  ret void
}

define void @stack_fold_cvtdq2ps(<128 x i32>* %a, <128 x i32>* %b, <128 x float>* %c) {
  ;CHECK-LABEL: stack_fold_cvtdq2ps
  ;CHECK:   vcvtdq2ps {{[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload

  %1 = load <128 x i32>* %a
  %2 = load <128 x i32>* %b
  %3 = and <128 x i32> %1, %2
  %4 = xor <128 x i32> %1, %2
  %5 = sitofp <128 x i32> %3 to <128 x float>
  %6 = sitofp <128 x i32> %4 to <128 x float>
  %7 = fadd <128 x float> %5, %6
  store <128 x float> %7, <128 x float>* %c
  ret void
}

define void @stack_fold_cvtpd2ps(<128 x double>* %a, <128 x double>* %b, <128 x float>* %c) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps
  ;CHECK:   vcvtpd2psy {{[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload

  %1 = load <128 x double>* %a
  %2 = load <128 x double>* %b
  %3 = fadd <128 x double> %1, %2
  %4 = fsub <128 x double> %1, %2
  %5 = fptrunc <128 x double> %3 to <128 x float>
  %6 = fptrunc <128 x double> %4 to <128 x float>
  %7 = fadd <128 x float> %5, %6
  store <128 x float> %7, <128 x float>* %c
  ret void
}

define void @stack_fold_cvttpd2dq(<64 x double>* %a, <64 x double>* %b, <64 x i32>* %c) #0 {
  ;CHECK-LABEL: stack_fold_cvttpd2dq
  ;CHECK:  vcvttpd2dqy {{[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload

  %1 = load <64 x double>* %a
  %2 = load <64 x double>* %b
  %3 = fadd <64 x double> %1, %2
  %4 = fsub <64 x double> %1, %2
  %5 = fptosi <64 x double> %3 to <64 x i32>
  %6 = fptosi <64 x double> %4 to <64 x i32>
  %7 = or <64 x i32> %5, %6
  store <64 x i32> %7, <64 x i32>* %c
  ret void
}

define void @stack_fold_cvttps2dq(<128 x float>* %a, <128 x float>* %b, <128 x i32>* %c) #0 {
  ;CHECK-LABEL: stack_fold_cvttps2dq
  ;CHECK:   vcvttps2dq {{[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload

  %1 = load <128 x float>* %a
  %2 = load <128 x float>* %b
  %3 = fadd <128 x float> %1, %2
  %4 = fsub <128 x float> %1, %2
  %5 = fptosi <128 x float> %3 to <128 x i32>
  %6 = fptosi <128 x float> %4 to <128 x i32>
  %7 = or <128 x i32> %5, %6
  store <128 x i32> %7, <128 x i32>* %c
  ret void
}
