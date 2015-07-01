; RUN: llc < %s -mtriple=i386-unknown-unknown -jump-is-expensive=0 | FileCheck %s --check-prefix=JUMP2
; RUN: llc < %s -mtriple=i386-unknown-unknown -jump-is-expensive=1 | FileCheck %s --check-prefix=JUMP1

define void @foo(i32 %X, i32 %Y, i32 %Z) nounwind {
; JUMP2-LABEL: foo:
; JUMP2-DAG:     jl
; JUMP2-DAG:     je
;
; JUMP1-LABEL: foo:
; JUMP1-DAG:     sete
; JUMP1-DAG:     setl
; JUMP1:         orb
; JUMP1:         jne
entry:
  %tmp1 = icmp eq i32 %X, 0
  %tmp3 = icmp slt i32 %Y, 5
  %tmp4 = or i1 %tmp3, %tmp1
  br i1 %tmp4, label %cond_true, label %UnifiedReturnBlock

cond_true:
  %tmp5 = tail call i32 (...) @bar( )
  ret void

UnifiedReturnBlock:
  ret void
}

declare i32 @bar(...)
