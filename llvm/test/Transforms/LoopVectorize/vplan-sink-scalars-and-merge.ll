; REQUIRES: asserts

; RUN: opt -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -debug -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@a = common global [2048 x i32] zeroinitializer, align 16
@b = common global [2048 x i32] zeroinitializer, align 16
@c = common global [2048 x i32] zeroinitializer, align 16


; CHECK-LABEL: LV: Checking a loop in "sink1"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   EMIT vp<%2> = icmp ule ir<%iv> vp<%0>
; CHECK-NEXT: Successor(s): loop.0

; CHECK:      loop.0:
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:      pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, ir<%iv>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b>
; CHECK-NEXT:     REPLICATE ir<%add> = add ir<%lv.b>, ir<10>
; CHECK-NEXT:     REPLICATE ir<%mul> = mul ir<2>, ir<%add>
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%iv>
; CHECK-NEXT:     REPLICATE store ir<%mul>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   PHI-PREDICATED-INSTRUCTION vp<%9> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.1:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
define void @sink1(i32 %k) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.b = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i32 0, i32 %iv
  %lv.b  = load i32, i32* %gep.b, align 4
  %add = add i32 %lv.b, 10
  %mul = mul i32 2, %add
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %iv
  store i32 %mul, i32* %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: LV: Checking a loop in "sink2"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   EMIT vp<%2> = icmp ule ir<%iv> vp<%0>
; CHECK-NEXT: Successor(s): pred.load

; CHECK:      <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:      pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, ir<%iv>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b>
; CHECK-NEXT:   Successor(s): pred.load.continue

; CHECK:      pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%5> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.0:
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%iv>, ir<2>
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:       pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%add> = add vp<%5>, ir<10>
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE store ir<%add>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:       loop.1:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
define void @sink2(i32 %k) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.b = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i32 0, i32 %iv
  %lv.b  = load i32, i32* %gep.b, align 4
  %add = add i32 %lv.b, 10
  %mul = mul i32 %iv, 2
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %mul
  store i32 %add, i32* %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: LV: Checking a loop in "sink3"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   EMIT vp<%2> = icmp ule ir<%iv> vp<%0>
; CHECK-NEXT: Successor(s): pred.load

; CHECK:      <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:       pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, ir<%iv>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue

; CHECK:       pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%5> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.0:
; CHECK-NEXT:   WIDEN ir<%add> = add vp<%5>, ir<10>
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%iv>, ir<%add>
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:      pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE store ir<%add>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.1:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
define void @sink3(i32 %k) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.b = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i32 0, i32 %iv
  %lv.b  = load i32, i32* %gep.b, align 4
  %add = add i32 %lv.b, 10
  %mul = mul i32 %iv, %add
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %mul
  store i32 %add, i32* %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; Make sure we do not sink uniform instructions.
define void @uniform_gep(i64 %k, i16* noalias %A, i16* noalias %B) {
; CHECK-LABEL: LV: Checking a loop in "uniform_gep"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 21, %iv.next
; CHECK-NEXT:   EMIT vp<%2> = WIDEN-CANONICAL-INDUCTION
; CHECK-NEXT:   EMIT vp<%3> = icmp ule vp<%2> vp<%0>
; CHECK-NEXT:   CLONE ir<%gep.A.uniform> = getelementptr ir<%A>, ir<0>
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%3>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<%3> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%lv> = load ir<%gep.A.uniform>
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%6> = ir<%lv>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT:   WIDEN ir<%cmp> = icmp ir<%iv>, ir<%k>
; CHECK-NEXT: Successor(s): loop.then
; CHECK-EMPTY:
; CHECK-NEXT: loop.then:
; CHECK-NEXT:   EMIT vp<%8> = not ir<%cmp>
; CHECK-NEXT:   EMIT vp<%9> = select vp<%3> vp<%8> ir<false>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%9>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%9> (loop.then)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.B> = getelementptr ir<%B>, ir<%iv>
; CHECK-NEXT:     REPLICATE store vp<%6>, ir<%gep.B>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.then.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.then.0:
; CHECK-NEXT: Successor(s): loop.latch
; CHECK-EMPTY:
; CHECK-NEXT: loop.latch:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 21, %entry ], [ %iv.next, %loop.latch ]
  %gep.A.uniform = getelementptr inbounds i16, i16* %A, i64 0
  %gep.B = getelementptr inbounds i16, i16* %B, i64 %iv
  %lv = load i16, i16* %gep.A.uniform, align 1
  %cmp = icmp ult i64 %iv, %k
  br i1 %cmp, label %loop.latch, label %loop.then

loop.then:
  store i16 %lv, i16* %gep.B, align 1
  br label %loop.latch

loop.latch:
  %iv.next = add nsw i64 %iv, 1
  %cmp179 = icmp slt i64 %iv.next, 32
  br i1 %cmp179, label %loop, label %exit
exit:
  ret void
}

; Loop with predicated load.
define void @pred_cfg1(i32 %k, i32 %j) {
; CHECK-LABEL: LV: Checking a loop in "pred_cfg1"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   WIDEN ir<%c.1> = icmp ir<%iv>, ir<%j>
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%iv>, ir<10>
; CHECK-NEXT: Successor(s): then.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0:
; CHECK-NEXT:   EMIT vp<%4> = icmp ule ir<%iv> vp<%0>
; CHECK-NEXT:   EMIT vp<%5> = select vp<%4> ir<%c.1> ir<false>
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%5>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<%5> (then.0)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, ir<%iv>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%8> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.0.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0.0:
; CHECK-NEXT: Successor(s): next.0
; CHECK-EMPTY:
; CHECK-NEXT: next.0:
; CHECK-NEXT:   EMIT vp<%9> = not ir<%c.1>
; CHECK-NEXT:   EMIT vp<%10> = select vp<%4> vp<%9> ir<false>
; CHECK-NEXT:   BLEND %p = ir<0>/vp<%10> vp<%8>/vp<%5>
; CHECK-NEXT:   EMIT vp<%12> = or vp<%5> vp<%10>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%12>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%12> (next.0)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE store ir<%p>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): next.0.0
; CHECK-EMPTY:
; CHECK-NEXT: next.0.0:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %next.0 ]
  %gep.b = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i32 0, i32 %iv
  %c.1 = icmp ult i32 %iv, %j
  %mul = mul i32 %iv, 10
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %mul
  br i1 %c.1, label %then.0, label %next.0

then.0:
  %lv.b  = load i32, i32* %gep.b, align 4
  br label %next.0

next.0:
  %p = phi i32 [ 0, %loop ], [ %lv.b, %then.0 ]
  store i32 %p, i32* %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; Loop with predicated load and store in separate blocks, store depends on
; loaded value.
define void @pred_cfg2(i32 %k, i32 %j) {
; CHECK-LABEL: LV: Checking a loop in "pred_cfg2"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%iv>, ir<10>
; CHECK-NEXT:   WIDEN ir<%c.0> = icmp ir<%iv>, ir<%j>
; CHECK-NEXT:   WIDEN ir<%c.1> = icmp ir<%iv>, ir<%j>
; CHECK-NEXT: Successor(s): then.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0:
; CHECK-NEXT:   EMIT vp<%5> = icmp ule ir<%iv> vp<%0>
; CHECK-NEXT:   EMIT vp<%6> = select vp<%5> ir<%c.0> ir<false>
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%6>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<%6> (then.0)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, ir<%iv>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%9> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.0.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0.0:
; CHECK-NEXT: Successor(s): next.0
; CHECK-EMPTY:
; CHECK-NEXT: next.0:
; CHECK-NEXT:   EMIT vp<%10> = not ir<%c.0>
; CHECK-NEXT:   EMIT vp<%11> = select vp<%5> vp<%10> ir<false>
; CHECK-NEXT:   BLEND %p = ir<0>/vp<%11> vp<%9>/vp<%6>
; CHECK-NEXT: Successor(s): then.1
; CHECK-EMPTY:
; CHECK-NEXT: then.1:
; CHECK-NEXT:   EMIT vp<%13> = or vp<%6> vp<%11>
; CHECK-NEXT:   EMIT vp<%14> = select vp<%13> ir<%c.1> ir<false>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%14>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%14> (then.1)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE store ir<%p>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.1.0
; CHECK-EMPTY:
; CHECK-NEXT: then.1.0:
; CHECK-NEXT: Successor(s): next.1
; CHECK-EMPTY:
; CHECK-NEXT: next.1:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %next.1 ]
  %gep.b = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i32 0, i32 %iv
  %mul = mul i32 %iv, 10
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %mul
  %c.0 = icmp ult i32 %iv, %j
  %c.1 = icmp ugt i32 %iv, %j
  br i1 %c.0, label %then.0, label %next.0

then.0:
  %lv.b  = load i32, i32* %gep.b, align 4
  br label %next.0

next.0:
  %p = phi i32 [ 0, %loop ], [ %lv.b, %then.0 ]
  br i1 %c.1, label %then.1, label %next.1

then.1:
  store i32 %p, i32* %gep.a, align 4
  br label %next.1

next.1:
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; Loop with predicated load and store in separate blocks, store does not depend
; on loaded value.
define void @pred_cfg3(i32 %k, i32 %j) {
; CHECK-LABEL: LV: Checking a loop in "pred_cfg3"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%iv>, ir<10>
; CHECK-NEXT:   WIDEN ir<%c.0> = icmp ir<%iv>, ir<%j>
; CHECK-NEXT: Successor(s): then.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0:
; CHECK-NEXT:   EMIT vp<%4> = icmp ule ir<%iv> vp<%0>
; CHECK-NEXT:   EMIT vp<%5> = select vp<%4> ir<%c.0> ir<false>
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%5>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<%5> (then.0)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, ir<%iv>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b>
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%8> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.0.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0.0:
; CHECK-NEXT: Successor(s): next.0
; CHECK-EMPTY:
; CHECK-NEXT: next.0:
; CHECK-NEXT: Successor(s): then.1
; CHECK-EMPTY:
; CHECK-NEXT: then.1:
; CHECK-NEXT:   EMIT vp<%9> = not ir<%c.0>
; CHECK-NEXT:   EMIT vp<%10> = select vp<%4> vp<%9> ir<false>
; CHECK-NEXT:   EMIT vp<%11> = or vp<%5> vp<%10>
; CHECK-NEXT:   EMIT vp<%12> = select vp<%11> ir<%c.0> ir<false>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%12>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%12> (then.1)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE store ir<0>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.1.0
; CHECK-EMPTY:
; CHECK-NEXT: then.1.0:
; CHECK-NEXT: Successor(s): next.1
; CHECK-EMPTY:
; CHECK-NEXT: next.1:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %next.1 ]
  %gep.b = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i32 0, i32 %iv
  %mul = mul i32 %iv, 10
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %mul
  %c.0 = icmp ult i32 %iv, %j
  br i1 %c.0, label %then.0, label %next.0

then.0:
  %lv.b  = load i32, i32* %gep.b, align 4
  br label %next.0

next.0:
  br i1 %c.0, label %then.1, label %next.1

then.1:
  store i32 0, i32* %gep.a, align 4
  br label %next.1

next.1:
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

define void @merge_3_replicate_region(i32 %k, i32 %j) {
; CHECK-LABEL: LV: Checking a loop in "merge_3_replicate_region"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   EMIT vp<%2> = icmp ule ir<%iv> vp<%0>
; CHECK-NEXT:   REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%iv>
; CHECK-NEXT: Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT: Successor(s): loop.1
; CHECK-EMPTY:
; CHECK-NEXT: loop.1:
; CHECK-NEXT: Successor(s): loop.2
; CHECK-EMPTY:
; CHECK-NEXT: loop.2:
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%lv.a> = load ir<%gep.a>
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, ir<%iv>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b>
; CHECK-NEXT:     REPLICATE ir<%gep.c> = getelementptr ir<@c>, ir<0>, ir<%iv>
; CHECK-NEXT:     REPLICATE store ir<%lv.a>, ir<%gep.c>
; CHECK-NEXT:     REPLICATE store ir<%lv.b>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%10> = ir<%lv.a>
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%11> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.3
; CHECK-EMPTY:
; CHECK-NEXT: loop.3:
; CHECK-NEXT:   WIDEN ir<%c.0> = icmp ir<%iv>, ir<%j>
; CHECK-NEXT: Successor(s): then.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0:
; CHECK-NEXT:   WIDEN ir<%mul> = mul vp<%10>, vp<%11>
; CHECK-NEXT:   EMIT vp<%14> = select vp<%2> ir<%c.0> ir<false>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%14>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%14> (then.0)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.c.1> = getelementptr ir<@c>, ir<0>, ir<%iv>
; CHECK-NEXT:     REPLICATE store ir<%mul>, ir<%gep.c.1>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.0.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0.0:
; CHECK-NEXT: Successor(s): latch
; CHECK-EMPTY:
; CHECK-NEXT: latch:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %iv
  %lv.a  = load i32, i32* %gep.a, align 4
  %gep.b = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i32 0, i32 %iv
  %lv.b  = load i32, i32* %gep.b, align 4
  %gep.c = getelementptr inbounds [2048 x i32], [2048 x i32]* @c, i32 0, i32 %iv
  store i32 %lv.a, i32* %gep.c, align 4
  store i32 %lv.b, i32* %gep.a, align 4
  %c.0 = icmp ult i32 %iv, %j
  br i1 %c.0, label %then.0, label %latch

then.0:
  %mul = mul i32 %lv.a, %lv.b
  %gep.c.1 = getelementptr inbounds [2048 x i32], [2048 x i32]* @c, i32 0, i32 %iv
  store i32 %mul, i32* %gep.c.1, align 4
  br label %latch

latch:
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}


define void @update_2_uses_in_same_recipe_in_merged_block(i32 %k) {
; CHECK-LABEL: LV: Checking a loop in "update_2_uses_in_same_recipe_in_merged_block"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   EMIT vp<%2> = icmp ule ir<%iv> vp<%0>
; CHECK-NEXT:   REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%iv>
; CHECK-NEXT: Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT: Successor(s): loop.1
; CHECK-EMPTY:
; CHECK-NEXT: loop.1:
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%lv.a> = load ir<%gep.a>
; CHECK-NEXT:     REPLICATE ir<%div> = sdiv ir<%lv.a>, ir<%lv.a>
; CHECK-NEXT:     REPLICATE store ir<%div>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%7> = ir<%lv.a>
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%8> = ir<%div>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.2
; CHECK-EMPTY:
; CHECK-NEXT: loop.2:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %iv
  %lv.a  = load i32, i32* %gep.a, align 4
  %div = sdiv i32 %lv.a, %lv.a
  store i32 %div, i32* %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

define void @recipe_in_merge_candidate_used_by_first_order_recurrence(i32 %k) {
; CHECK-LABEL: LV: Checking a loop in "recipe_in_merge_candidate_used_by_first_order_recurrence"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   FIRST-ORDER-RECURRENCE-PHI ir<%for> = phi ir<0>, ir<%lv.a>
; CHECK-NEXT:   EMIT vp<%3> = icmp ule ir<%iv> vp<%0>
; CHECK-NEXT:   REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%iv>
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%3>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<%3> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%lv.a> = load ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%6> = ir<%lv.a>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT:   EMIT vp<%7> = first-order splice ir<%for> ir<%lv.a>
; CHECK-NEXT: Successor(s): loop.1
; CHECK-EMPTY:
; CHECK-NEXT: loop.1:
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%3>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%3> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%div> = sdiv vp<%7>, vp<%6>
; CHECK-NEXT:     REPLICATE store ir<%div>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%10> = ir<%div>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.2
; CHECK-EMPTY:
; CHECK-NEXT: loop.2:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %for = phi i32 [ 0, %entry ], [ %lv.a, %loop ]
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %iv
  %lv.a  = load i32, i32* %gep.a, align 4
  %div = sdiv i32 %for, %lv.a
  store i32 %div, i32* %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}
