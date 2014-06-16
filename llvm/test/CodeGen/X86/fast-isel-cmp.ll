; RUN: llc < %s                             -mtriple=x86_64-apple-darwin10 | FileCheck %s --check-prefix=SDAG
; RUN: llc < %s -fast-isel -fast-isel-abort -mtriple=x86_64-apple-darwin10 | FileCheck %s --check-prefix=FAST

define zeroext i1 @fcmp_oeq(float %x, float %y) {
; SDAG-LABEL: fcmp_oeq
; SDAG:       cmpeqss  %xmm1, %xmm0
; SDAG-NEXT:  movd     %xmm0, %eax
; SDAG-NEXT:  andl     $1, %eax
; FAST-LABEL: fcmp_oeq
; FAST:       ucomiss  %xmm1, %xmm0
; FAST-NEXT:  sete     %al
; FAST-NEXT:  setnp    %cl
; FAST-NEXT:  andb     %al, %cl
  %1 = fcmp oeq float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_ogt(float %x, float %y) {
; SDAG-LABEL: fcmp_ogt
; SDAG:       ucomiss  %xmm1, %xmm0
; SDAG-NEXT:  seta     %al
; FAST:       ucomiss  %xmm1, %xmm0
; FAST-NEXT:  seta     %al
  %1 = fcmp ogt float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_oge(float %x, float %y) {
; SDAG-LABEL: fcmp_oge
; SDAG:       ucomiss  %xmm1, %xmm0
; SDAG-NEXT:  setae    %al
; FAST-LABEL: fcmp_oge
; FAST:       ucomiss  %xmm1, %xmm0
; FAST-NEXT:  setae    %al
  %1 = fcmp oge float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_olt(float %x, float %y) {
; SDAG-LABEL: fcmp_olt
; SDAG:       ucomiss  %xmm0, %xmm1
; SDAG-NEXT:  seta     %al
; FAST-LABEL: fcmp_olt
; FAST:       ucomiss  %xmm0, %xmm1
; FAST-NEXT:  seta     %al
  %1 = fcmp olt float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_ole(float %x, float %y) {
; SDAG-LABEL: fcmp_ole
; SDAG:       ucomiss  %xmm0, %xmm1
; SDAG-NEXT:  setae    %al
; FAST-LABEL: fcmp_ole
; FAST:       ucomiss  %xmm0, %xmm1
; FAST-NEXT:  setae    %al
  %1 = fcmp ole float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_one(float %x, float %y) {
; SDAG-LABEL: fcmp_one
; SDAG:       ucomiss  %xmm1, %xmm0
; SDAG-NEXT:  setne    %al
; FAST-LABEL: fcmp_one
; FAST:       ucomiss  %xmm1, %xmm0
; FAST-NEXT:  setne    %al
  %1 = fcmp one float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_ord(float %x, float %y) {
; SDAG-LABEL: fcmp_ord
; SDAG:       ucomiss  %xmm1, %xmm0
; SDAG-NEXT:  setnp    %al
; FAST-LABEL: fcmp_ord
; FAST:       ucomiss  %xmm1, %xmm0
; FAST-NEXT:  setnp    %al
  %1 = fcmp ord float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_uno(float %x, float %y) {
; SDAG-LABEL: fcmp_uno
; SDAG:       ucomiss  %xmm1, %xmm0
; SDAG-NEXT:  setp     %al
; FAST-LABEL: fcmp_uno
; FAST:       ucomiss  %xmm1, %xmm0
; FAST-NEXT:  setp     %al
  %1 = fcmp uno float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_ueq(float %x, float %y) {
; SDAG-LABEL: fcmp_ueq
; SDAG:       ucomiss  %xmm1, %xmm0
; SDAG-NEXT:  sete     %al
; FAST-LABEL: fcmp_ueq
; FAST:       ucomiss  %xmm1, %xmm0
; FAST-NEXT:  sete     %al
  %1 = fcmp ueq float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_ugt(float %x, float %y) {
; SDAG-LABEL: fcmp_ugt
; SDAG:       ucomiss  %xmm0, %xmm1
; SDAG-NEXT:  setb     %al
; FAST-LABEL: fcmp_ugt
; FAST:       ucomiss  %xmm0, %xmm1
; FAST-NEXT:  setb     %al
  %1 = fcmp ugt float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_uge(float %x, float %y) {
; SDAG-LABEL: fcmp_uge
; SDAG:       ucomiss  %xmm0, %xmm1
; SDAG-NEXT:  setbe    %al
; FAST-LABEL: fcmp_uge
; FAST:       ucomiss  %xmm0, %xmm1
; FAST-NEXT:  setbe    %al
  %1 = fcmp uge float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_ult(float %x, float %y) {
; SDAG-LABEL: fcmp_ult
; SDAG:       ucomiss  %xmm1, %xmm0
; SDAG-NEXT:  setb     %al
; FAST-LABEL: fcmp_ult
; FAST:       ucomiss  %xmm1, %xmm0
; FAST-NEXT:  setb     %al
  %1 = fcmp ult float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_ule(float %x, float %y) {
; SDAG-LABEL: fcmp_ule
; SDAG:       ucomiss  %xmm1, %xmm0
; SDAG-NEXT:  setbe    %al
; FAST-LABEL: fcmp_ule
; FAST:       ucomiss  %xmm1, %xmm0
; FAST-NEXT:  setbe    %al
  %1 = fcmp ule float %x, %y
  ret i1 %1
}

define zeroext i1 @fcmp_une(float %x, float %y) {
; SDAG-LABEL: fcmp_une
; SDAG:       cmpneqss %xmm1, %xmm0
; SDAG-NEXT:  movd     %xmm0, %eax
; SDAG-NEXT:  andl     $1, %eax
; FAST-LABEL: fcmp_une
; FAST:       ucomiss  %xmm1, %xmm0
; FAST-NEXT:  setne    %al
; FAST-NEXT:  setp     %cl
; FAST-NEXT:  andb     %al, %cl
  %1 = fcmp une float %x, %y
  ret i1 %1
}

define zeroext i1 @icmp_eq(i32 %x, i32 %y) {
; SDAG-LABEL: icmp_eq
; SDAG:       cmpl     %esi, %edi
; SDAG-NEXT:  sete     %al
; FAST-LABEL: icmp_eq
; FAST:       cmpl     %esi, %edi
; FAST-NEXT:  sete     %al
  %1 = icmp eq i32 %x, %y
  ret i1 %1
}

define zeroext i1 @icmp_ne(i32 %x, i32 %y) {
; SDAG-LABEL: icmp_ne
; SDAG:       cmpl     %esi, %edi
; SDAG-NEXT:  setne    %al
; FAST-LABEL: icmp_ne
; FAST:       cmpl     %esi, %edi
; FAST-NEXT:  setne    %al
  %1 = icmp ne i32 %x, %y
  ret i1 %1
}

define zeroext i1 @icmp_ugt(i32 %x, i32 %y) {
; SDAG-LABEL: icmp_ugt
; SDAG:       cmpl     %edi, %esi
; SDAG-NEXT:  setb     %al
; FAST-LABEL: icmp_ugt
; FAST:       cmpl     %esi, %edi
; FAST-NEXT:  seta     %al
  %1 = icmp ugt i32 %x, %y
  ret i1 %1
}

define zeroext i1 @icmp_uge(i32 %x, i32 %y) {
; SDAG-LABEL: icmp_uge
; SDAG:       cmpl     %esi, %edi
; SDAG-NEXT:  setae    %al
; FAST-LABEL: icmp_uge
; FAST:       cmpl     %esi, %edi
; FAST-NEXT:  setae    %al
  %1 = icmp uge i32 %x, %y
  ret i1 %1
}

define zeroext i1 @icmp_ult(i32 %x, i32 %y) {
; SDAG-LABEL: icmp_ult
; SDAG:       cmpl     %esi, %edi
; SDAG-NEXT:  setb     %al
; FAST-LABEL: icmp_ult
; FAST:       cmpl     %esi, %edi
; FAST-NEXT:  setb     %al
  %1 = icmp ult i32 %x, %y
  ret i1 %1
}

define zeroext i1 @icmp_ule(i32 %x, i32 %y) {
; SDAG-LABEL: icmp_ule
; SDAG:       cmpl     %esi, %edi
; SDAG-NEXT:  setbe    %al
; FAST-LABEL: icmp_ule
; FAST:       cmpl     %esi, %edi
; FAST-NEXT:  setbe    %al
  %1 = icmp ule i32 %x, %y
  ret i1 %1
}

define zeroext i1 @icmp_sgt(i32 %x, i32 %y) {
; SDAG-LABEL: icmp_sgt
; SDAG:       cmpl     %esi, %edi
; SDAG-NEXT:  setg     %al
; FAST-LABEL: icmp_sgt
; FAST:       cmpl     %esi, %edi
; FAST-NEXT:  setg     %al
  %1 = icmp sgt i32 %x, %y
  ret i1 %1
}

define zeroext i1 @icmp_sge(i32 %x, i32 %y) {
; SDAG-LABEL: icmp_sge
; SDAG:       cmpl     %esi, %edi
; SDAG-NEXT:  setge    %al
; FAST-LABEL: icmp_sge
; FAST:       cmpl     %esi, %edi
; FAST-NEXT:  setge    %al
  %1 = icmp sge i32 %x, %y
  ret i1 %1
}

define zeroext i1 @icmp_slt(i32 %x, i32 %y) {
; SDAG-LABEL: icmp_slt
; SDAG:       cmpl     %esi, %edi
; SDAG-NEXT:  setl     %al
; FAST-LABEL: icmp_slt
; FAST:       cmpl     %esi, %edi
; FAST-NEXT:  setl     %al
  %1 = icmp slt i32 %x, %y
  ret i1 %1
}

define zeroext i1 @icmp_sle(i32 %x, i32 %y) {
; SDAG-LABEL: icmp_sle
; SDAG:       cmpl     %esi, %edi
; SDAG-NEXT:  setle    %al
; FAST-LABEL: icmp_sle
; FAST:       cmpl     %esi, %edi
; FAST-NEXT:  setle    %al
  %1 = icmp sle i32 %x, %y
  ret i1 %1
}

