; RUN: llc < %s -mtriple=x86_64-apple-darwin10                             | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -fast-isel -fast-isel-abort=1 | FileCheck %s

; Test all the cmp predicates that can feed an integer conditional move.

define i64 @select_fcmp_false_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_false_cmov
; CHECK:       movq %rsi, %rax
; CHECK-NEXT:  retq
  %1 = fcmp false double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_oeq_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_oeq_cmov
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  setnp %al
; CHECK-NEXT:  sete %cl
; CHECK-NEXT:  testb %al, %cl
; CHECK-NEXT:  cmoveq %rsi, %rdi
  %1 = fcmp oeq double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_ogt_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_ogt_cmov
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  cmovbeq %rsi, %rdi
  %1 = fcmp ogt double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_oge_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_oge_cmov
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  cmovbq %rsi, %rdi
  %1 = fcmp oge double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_olt_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_olt_cmov
; CHECK:       ucomisd %xmm0, %xmm1
; CHECK-NEXT:  cmovbeq %rsi, %rdi
  %1 = fcmp olt double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_ole_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_ole_cmov
; CHECK:       ucomisd %xmm0, %xmm1
; CHECK-NEXT:  cmovbq %rsi, %rdi
  %1 = fcmp ole double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_one_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_one_cmov
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  cmoveq %rsi, %rdi
  %1 = fcmp one double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_ord_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_ord_cmov
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  cmovpq %rsi, %rdi
  %1 = fcmp ord double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_uno_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_uno_cmov
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  cmovnpq %rsi, %rdi
  %1 = fcmp uno double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_ueq_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_ueq_cmov
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  cmovneq %rsi, %rdi
  %1 = fcmp ueq double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_ugt_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_ugt_cmov
; CHECK:       ucomisd %xmm0, %xmm1
; CHECK-NEXT:  cmovaeq %rsi, %rdi
  %1 = fcmp ugt double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_uge_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_uge_cmov
; CHECK:       ucomisd %xmm0, %xmm1
; CHECK-NEXT:  cmovaq %rsi, %rdi
  %1 = fcmp uge double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_ult_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_ult_cmov
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  cmovaeq %rsi, %rdi
  %1 = fcmp ult double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_ule_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_ule_cmov
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  cmovaq %rsi, %rdi
  %1 = fcmp ule double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_une_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_une_cmov
; CHECK:       ucomisd %xmm1, %xmm0
; CHECK-NEXT:  setp %al
; CHECK-NEXT:  setne %cl
; CHECK-NEXT:  orb %al, %cl
; CHECK-NEXT:  cmoveq %rsi, %rdi
  %1 = fcmp une double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_fcmp_true_cmov(double %a, double %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_fcmp_true_cmov
; CHECK:       movq %rdi, %rax
  %1 = fcmp true double %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_icmp_eq_cmov(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_icmp_eq_cmov
; CHECK:       cmpq    %rsi, %rdi
; CHECK-NEXT:  cmovneq %rcx, %rdx
; CHECK-NEXT:  movq    %rdx, %rax
  %1 = icmp eq i64 %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_icmp_ne_cmov(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_icmp_ne_cmov
; CHECK:       cmpq    %rsi, %rdi
; CHECK-NEXT:  cmoveq  %rcx, %rdx
; CHECK-NEXT:  movq    %rdx, %rax
  %1 = icmp ne i64 %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_icmp_ugt_cmov(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_icmp_ugt_cmov
; CHECK:       cmpq    %rsi, %rdi
; CHECK-NEXT:  cmovbeq %rcx, %rdx
; CHECK-NEXT:  movq    %rdx, %rax
  %1 = icmp ugt i64 %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}


define i64 @select_icmp_uge_cmov(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_icmp_uge_cmov
; CHECK:       cmpq    %rsi, %rdi
; CHECK-NEXT:  cmovbq  %rcx, %rdx
; CHECK-NEXT:  movq    %rdx, %rax
  %1 = icmp uge i64 %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_icmp_ult_cmov(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_icmp_ult_cmov
; CHECK:       cmpq    %rsi, %rdi
; CHECK-NEXT:  cmovaeq %rcx, %rdx
; CHECK-NEXT:  movq    %rdx, %rax
  %1 = icmp ult i64 %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_icmp_ule_cmov(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_icmp_ule_cmov
; CHECK:       cmpq    %rsi, %rdi
; CHECK-NEXT:  cmovaq  %rcx, %rdx
; CHECK-NEXT:  movq    %rdx, %rax
  %1 = icmp ule i64 %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_icmp_sgt_cmov(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_icmp_sgt_cmov
; CHECK:       cmpq    %rsi, %rdi
; CHECK-NEXT:  cmovleq %rcx, %rdx
; CHECK-NEXT:  movq    %rdx, %rax
  %1 = icmp sgt i64 %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_icmp_sge_cmov(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_icmp_sge_cmov
; CHECK:       cmpq    %rsi, %rdi
; CHECK-NEXT:  cmovlq  %rcx, %rdx
; CHECK-NEXT:  movq    %rdx, %rax
  %1 = icmp sge i64 %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_icmp_slt_cmov(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_icmp_slt_cmov
; CHECK:       cmpq    %rsi, %rdi
; CHECK-NEXT:  cmovgeq %rcx, %rdx
; CHECK-NEXT:  movq    %rdx, %rax
  %1 = icmp slt i64 %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

define i64 @select_icmp_sle_cmov(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: select_icmp_sle_cmov
; CHECK:       cmpq    %rsi, %rdi
; CHECK-NEXT:  cmovgq  %rcx, %rdx
; CHECK-NEXT:  movq    %rdx, %rax
  %1 = icmp sle i64 %a, %b
  %2 = select i1 %1, i64 %c, i64 %d
  ret i64 %2
}

