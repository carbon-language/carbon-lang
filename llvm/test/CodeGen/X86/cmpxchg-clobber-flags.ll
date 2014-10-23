; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu -pre-RA-sched=fast %s -o - | FileCheck %s

declare i32 @bar()

define i64 @test_intervening_call(i64* %foo, i64 %bar, i64 %baz) {
; CHECK-LABEL: test_intervening_call:
; CHECK: cmpxchg
; CHECK: pushfq
; CHECK: popq [[FLAGS:%.*]]

; CHECK: callq bar

; CHECK: pushq [[FLAGS]]
; CHECK: popfq
; CHECK: jne
  %cx = cmpxchg i64* %foo, i64 %bar, i64 %baz seq_cst seq_cst
  %p = extractvalue { i64, i1 } %cx, 1
  call i32 @bar()
  br i1 %p, label %t, label %f

t:
  ret i64 42

f:
  ret i64 0
}

; Interesting in producing a clobber without any function calls.
define i32 @test_control_flow(i32* %p, i32 %i, i32 %j) {
; CHECK-LABEL: test_control_flow:

; CHECK: cmpxchg
; CHECK-NEXT: jne
entry:
  %cmp = icmp sgt i32 %i, %j
  br i1 %cmp, label %loop_start, label %cond.end

loop_start:
  br label %while.condthread-pre-split.i

while.condthread-pre-split.i:
  %.pr.i = load i32* %p, align 4
  br label %while.cond.i

while.cond.i:
  %0 = phi i32 [ %.pr.i, %while.condthread-pre-split.i ], [ 0, %while.cond.i ]
  %tobool.i = icmp eq i32 %0, 0
  br i1 %tobool.i, label %while.cond.i, label %while.body.i

while.body.i:
  %.lcssa = phi i32 [ %0, %while.cond.i ]
  %1 = cmpxchg i32* %p, i32 %.lcssa, i32 %.lcssa seq_cst seq_cst
  %2 = extractvalue { i32, i1 } %1, 1
  br i1 %2, label %cond.end.loopexit, label %while.condthread-pre-split.i

cond.end.loopexit:
  br label %cond.end

cond.end:
  %cond = phi i32 [ %i, %entry ], [ 0, %cond.end.loopexit ]
  ret i32 %cond
}

; This one is an interesting case because CMOV doesn't have a chain
; operand. Naive attempts to limit cmpxchg EFLAGS use are likely to fail here.
define i32 @test_feed_cmov(i32* %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: test_feed_cmov:

; CHECK: cmpxchg
; CHECK: pushfq
; CHECK: popq [[FLAGS:%.*]]

; CHECK: callq bar

; CHECK: pushq [[FLAGS]]
; CHECK: popfq

  %res = cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  %success = extractvalue { i32, i1 } %res, 1

  %rhs = call i32 @bar()

  %ret = select i1 %success, i32 %new, i32 %rhs
  ret i32 %ret
}
