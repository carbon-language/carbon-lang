; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s

; The comparison uses the pre-inc value, which could lead LSR to
; try to compute -INT64_MIN.

; CHECK: movabsq $-9223372036854775808, %rax
; CHECK: cmpq  %rax,
; CHECK: sete  %al

declare i64 @bar()

define i1 @foo() nounwind {
entry:
  br label %for.cond.i

for.cond.i:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.cond.i ]
  %t = call i64 @bar()
  %indvar.next = add i64 %indvar, 1
  %s = icmp ne i64 %indvar.next, %t
  br i1 %s, label %for.cond.i, label %__ABContainsLabel.exit

__ABContainsLabel.exit:
  %cmp = icmp eq i64 %indvar, 9223372036854775807
  ret i1 %cmp
}
