; RUN: llc -verify-machineinstrs -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -O0 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 < %s | FileCheck %s

; The second run of the test case is to ensure the behaviour is the same
; without specifying -mcpu=pwr8 as that is now the baseline for ppc64le.

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@number64 = global i64 10, align 8

; CHECK: .abiversion 2

define i64 @use_toc(i64 %a) nounwind {
entry:
; CHECK-LABEL: @use_toc
; CHECK-NEXT: .L{{.*}}:
; CHECK-NEXT: .Lfunc_gep[[FN:[0-9]+]]:
; CHECK-NEXT: addis 2, 12, .TOC.-.Lfunc_gep[[FN]]@ha
; CHECK-NEXT: addi 2, 2, .TOC.-.Lfunc_gep[[FN]]@l
; CHECK-NEXT: .Lfunc_lep[[FN]]:
; CHECK-NEXT: .localentry use_toc, .Lfunc_lep[[FN]]-.Lfunc_gep[[FN]]
; CHECK-NEXT: %entry
  %0 = load i64, i64* @number64, align 8
  %cmp = icmp eq i64 %0, %a
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

declare void @callee()
define void @use_toc_implicit() nounwind {
entry:
; CHECK-LABEL: @use_toc_implicit
; CHECK-NEXT: .L{{.*}}:
; CHECK-NEXT: .Lfunc_gep[[FN:[0-9]+]]:
; CHECK-NEXT: addis 2, 12, .TOC.-.Lfunc_gep[[FN]]@ha
; CHECK-NEXT: addi 2, 2, .TOC.-.Lfunc_gep[[FN]]@l
; CHECK-NEXT: .Lfunc_lep[[FN]]:
; CHECK-NEXT: .localentry use_toc_implicit, .Lfunc_lep[[FN]]-.Lfunc_gep[[FN]]
; CHECK-NEXT: %entry
  call void @callee()
  ret void
}

define i64 @no_toc(i64 %a) nounwind {
entry:
; CHECK-LABEL: @no_toc
; CHECK-NEXT: .L{{.*}}:
; CHECK-NEXT: %entry
  ret i64 %a
}

