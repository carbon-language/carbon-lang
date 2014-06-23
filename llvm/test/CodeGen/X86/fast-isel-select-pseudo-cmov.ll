; RUN: llc < %s -mtriple=x86_64-apple-darwin10                                              | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -fast-isel -fast-isel-abort                  | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10                             -mcpu=corei7-avx | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -fast-isel -fast-isel-abort -mcpu=corei7-avx | FileCheck %s


define float @select_fcmp_one_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_one_f32
; CHECK:       ucomiss %xmm1, %xmm0
; CHECK-NEXT:  jne [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  movaps %xmm2, %xmm0
  %1 = fcmp one float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_one_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_one_f64
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  jne [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  movaps  %xmm2, %xmm0
  %1 = fcmp one double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_icmp_eq_f32(i64 %a, i64 %b, float %c, float %d) {
; CHECK-LABEL: select_icmp_eq_f32
; CHECK:       cmpq %rsi, %rdi
; CHECK-NEXT:  je [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  retq
  %1 = icmp eq i64 %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define float @select_icmp_ne_f32(i64 %a, i64 %b, float %c, float %d) {
; CHECK-LABEL: select_icmp_ne_f32
; CHECK:       cmpq %rsi, %rdi
; CHECK-NEXT:  jne [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  retq
  %1 = icmp ne i64 %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define float @select_icmp_ugt_f32(i64 %a, i64 %b, float %c, float %d) {
; CHECK-LABEL: select_icmp_ugt_f32
; CHECK:       cmpq %rsi, %rdi
; CHECK-NEXT:  ja [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  retq
  %1 = icmp ugt i64 %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define float @select_icmp_uge_f32(i64 %a, i64 %b, float %c, float %d) {
; CHECK-LABEL: select_icmp_uge_f32
; CHECK:       cmpq %rsi, %rdi
; CHECK-NEXT:  jae [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  retq
  %1 = icmp uge i64 %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define float @select_icmp_ult_f32(i64 %a, i64 %b, float %c, float %d) {
; CHECK-LABEL: select_icmp_ult_f32
; CHECK:       cmpq %rsi, %rdi
; CHECK-NEXT:  jb [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  retq
  %1 = icmp ult i64 %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define float @select_icmp_ule_f32(i64 %a, i64 %b, float %c, float %d) {
; CHECK-LABEL: select_icmp_ule_f32
; CHECK:       cmpq %rsi, %rdi
; CHECK-NEXT:  jbe [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  retq
  %1 = icmp ule i64 %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define float @select_icmp_sgt_f32(i64 %a, i64 %b, float %c, float %d) {
; CHECK-LABEL: select_icmp_sgt_f32
; CHECK:       cmpq %rsi, %rdi
; CHECK-NEXT:  jg [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  retq
  %1 = icmp sgt i64 %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define float @select_icmp_sge_f32(i64 %a, i64 %b, float %c, float %d) {
; CHECK-LABEL: select_icmp_sge_f32
; CHECK:       cmpq %rsi, %rdi
; CHECK-NEXT:  jge [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  retq
  %1 = icmp sge i64 %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define float @select_icmp_slt_f32(i64 %a, i64 %b, float %c, float %d) {
; CHECK-LABEL: select_icmp_slt_f32
; CHECK:       cmpq %rsi, %rdi
; CHECK-NEXT:  jl [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  retq
  %1 = icmp slt i64 %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define float @select_icmp_sle_f32(i64 %a, i64 %b, float %c, float %d) {
; CHECK-LABEL: select_icmp_sle_f32
; CHECK:       cmpq %rsi, %rdi
; CHECK-NEXT:  jle [[BB:LBB[0-9]+_2]]
; CHECK:       [[BB]]
; CHECK-NEXT:  retq
  %1 = icmp sle i64 %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

