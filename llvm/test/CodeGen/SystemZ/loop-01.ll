; Test loop tuning.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Test that strength reduction is applied to addresses with a scale factor,
; but that indexed addressing can still be used.
define void @f1(i32 *%dest, i32 %a) {
; CHECK-LABEL: f1:
; CHECK-NOT: sllg
; CHECK: st %r3, 0({{%r[1-5],%r[1-5]}})
; CHECK: br %r14
entry:
  br label %loop

loop:
  %index = phi i64 [ 0, %entry ], [ %next, %loop ]
  %ptr = getelementptr i32, i32 *%dest, i64 %index
  store i32 %a, i32 *%ptr
  %next = add i64 %index, 1
  %cmp = icmp ne i64 %next, 100
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Test a loop that should be converted into dbr form and then use BRCT.
define void @f2(i32 *%src, i32 *%dest) {
; CHECK-LABEL: f2:
; CHECK: lhi [[REG:%r[0-5]]], 100
; CHECK: [[LABEL:\.[^:]*]]:{{.*}} %loop
; CHECK: brct [[REG]], [[LABEL]]
; CHECK: br %r14
entry:
  br label %loop

loop:
  %count = phi i32 [ 0, %entry ], [ %next, %loop.next ]
  %next = add i32 %count, 1
  %val = load volatile i32 , i32 *%src
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %loop.next, label %loop.store

loop.store:
  %add = add i32 %val, 1
  store volatile i32 %add, i32 *%dest
  br label %loop.next

loop.next:
  %cont = icmp ne i32 %next, 100
  br i1 %cont, label %loop, label %exit

exit:
  ret void
}

; Like f2, but for BRCTG.
define void @f3(i64 *%src, i64 *%dest) {
; CHECK-LABEL: f3:
; CHECK: lghi [[REG:%r[0-5]]], 100
; CHECK: [[LABEL:\.[^:]*]]:{{.*}} %loop
; CHECK: brctg [[REG]], [[LABEL]]
; CHECK: br %r14
entry:
  br label %loop

loop:
  %count = phi i64 [ 0, %entry ], [ %next, %loop.next ]
  %next = add i64 %count, 1
  %val = load volatile i64 , i64 *%src
  %cmp = icmp eq i64 %val, 0
  br i1 %cmp, label %loop.next, label %loop.store

loop.store:
  %add = add i64 %val, 1
  store volatile i64 %add, i64 *%dest
  br label %loop.next

loop.next:
  %cont = icmp ne i64 %next, 100
  br i1 %cont, label %loop, label %exit

exit:
  ret void
}

; Test a loop with a 64-bit decremented counter in which the 32-bit
; low part of the counter is used after the decrement.  This is an example
; of a subregister use being the only thing that blocks a conversion to BRCTG.
define void @f4(i32 *%src, i32 *%dest, i64 *%dest2, i64 %count) {
; CHECK-LABEL: f4:
; CHECK: aghi [[REG:%r[0-5]]], -1
; CHECK: lr [[REG2:%r[0-5]]], [[REG]]
; CHECK: stg [[REG2]],
; CHECK: jne {{\..*}}
; CHECK: br %r14
entry:
  br label %loop

loop:
  %left = phi i64 [ %count, %entry ], [ %next, %loop.next ]
  store volatile i64 %left, i64 *%dest2
  %val = load volatile i32 , i32 *%src
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %loop.next, label %loop.store

loop.store:
  %add = add i32 %val, 1
  store volatile i32 %add, i32 *%dest
  br label %loop.next

loop.next:
  %next = add i64 %left, -1
  %ext = zext i32 %val to i64
  %shl = shl i64 %ext, 32
  %and = and i64 %next, 4294967295
  %or = or i64 %shl, %and
  store volatile i64 %or, i64 *%dest2
  %cont = icmp ne i64 %next, 0
  br i1 %cont, label %loop, label %exit

exit:
  ret void
}
