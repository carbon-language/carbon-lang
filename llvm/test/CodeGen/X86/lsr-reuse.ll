; RUN: llc < %s -march=x86-64 -O3 -asm-verbose=false | FileCheck %s
target datalayout = "e-p:64:64:64"
target triple = "x86_64-unknown-unknown"

; Full strength reduction reduces register pressure from 5 to 4 here.
; Instruction selection should use the FLAGS value from the dec for
; the branch. Scheduling should push the adds upwards.

; CHECK: full_me_0:
; CHECK: movsd   (%rsi), %xmm0
; CHECK: mulsd   (%rdx), %xmm0
; CHECK: movsd   %xmm0, (%rdi)
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: addq    $8, %rdi
; CHECK: decq    %rcx
; CHECK: jne

define void @full_me_0(double* nocapture %A, double* nocapture %B, double* nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %Ai = getelementptr inbounds double* %A, i64 %i
  %Bi = getelementptr inbounds double* %B, i64 %i
  %Ci = getelementptr inbounds double* %C, i64 %i
  %t1 = load double* %Bi
  %t2 = load double* %Ci
  %m = fmul double %t1, %t2
  store double %m, double* %Ai
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; Mostly-full strength reduction means we do full strength reduction on all
; except for the offsets.
;
; Given a choice between constant offsets -2048 and 2048, choose the negative
; value, because at boundary conditions it has a smaller encoding.
; TODO: That's an over-general heuristic. It would be better for the target
; to indicate what the encoding cost would be. Then using a 2048 offset
; would be better on x86-64, since the start value would be 0 instead of
; 2048.

; CHECK: mostly_full_me_0:
; CHECK: movsd   -2048(%rsi), %xmm0
; CHECK: mulsd   -2048(%rdx), %xmm0
; CHECK: movsd   %xmm0, -2048(%rdi)
; CHECK: movsd   (%rsi), %xmm0
; CHECK: divsd   (%rdx), %xmm0
; CHECK: movsd   %xmm0, (%rdi)
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: addq    $8, %rdi
; CHECK: decq    %rcx
; CHECK: jne

define void @mostly_full_me_0(double* nocapture %A, double* nocapture %B, double* nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %Ai = getelementptr inbounds double* %A, i64 %i
  %Bi = getelementptr inbounds double* %B, i64 %i
  %Ci = getelementptr inbounds double* %C, i64 %i
  %t1 = load double* %Bi
  %t2 = load double* %Ci
  %m = fmul double %t1, %t2
  store double %m, double* %Ai
  %j = add i64 %i, 256
  %Aj = getelementptr inbounds double* %A, i64 %j
  %Bj = getelementptr inbounds double* %B, i64 %j
  %Cj = getelementptr inbounds double* %C, i64 %j
  %t3 = load double* %Bj
  %t4 = load double* %Cj
  %o = fdiv double %t3, %t4
  store double %o, double* %Aj
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; A minor variation on mostly_full_me_0.
; Prefer to start the indvar at 0.

; CHECK: mostly_full_me_1:
; CHECK: movsd   (%rsi), %xmm0
; CHECK: mulsd   (%rdx), %xmm0
; CHECK: movsd   %xmm0, (%rdi)
; CHECK: movsd   -2048(%rsi), %xmm0
; CHECK: divsd   -2048(%rdx), %xmm0
; CHECK: movsd   %xmm0, -2048(%rdi)
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: addq    $8, %rdi
; CHECK: decq    %rcx
; CHECK: jne

define void @mostly_full_me_1(double* nocapture %A, double* nocapture %B, double* nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %Ai = getelementptr inbounds double* %A, i64 %i
  %Bi = getelementptr inbounds double* %B, i64 %i
  %Ci = getelementptr inbounds double* %C, i64 %i
  %t1 = load double* %Bi
  %t2 = load double* %Ci
  %m = fmul double %t1, %t2
  store double %m, double* %Ai
  %j = sub i64 %i, 256
  %Aj = getelementptr inbounds double* %A, i64 %j
  %Bj = getelementptr inbounds double* %B, i64 %j
  %Cj = getelementptr inbounds double* %C, i64 %j
  %t3 = load double* %Bj
  %t4 = load double* %Cj
  %o = fdiv double %t3, %t4
  store double %o, double* %Aj
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; A slightly less minor variation on mostly_full_me_0.

; CHECK: mostly_full_me_2:
; CHECK: movsd   (%rsi), %xmm0
; CHECK: mulsd   (%rdx), %xmm0
; CHECK: movsd   %xmm0, (%rdi)
; CHECK: movsd   -4096(%rsi), %xmm0
; CHECK: divsd   -4096(%rdx), %xmm0
; CHECK: movsd   %xmm0, -4096(%rdi)
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: addq    $8, %rdi
; CHECK: decq    %rcx
; CHECK: jne

define void @mostly_full_me_2(double* nocapture %A, double* nocapture %B, double* nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %k = add i64 %i, 256
  %Ak = getelementptr inbounds double* %A, i64 %k
  %Bk = getelementptr inbounds double* %B, i64 %k
  %Ck = getelementptr inbounds double* %C, i64 %k
  %t1 = load double* %Bk
  %t2 = load double* %Ck
  %m = fmul double %t1, %t2
  store double %m, double* %Ak
  %j = sub i64 %i, 256
  %Aj = getelementptr inbounds double* %A, i64 %j
  %Bj = getelementptr inbounds double* %B, i64 %j
  %Cj = getelementptr inbounds double* %C, i64 %j
  %t3 = load double* %Bj
  %t4 = load double* %Cj
  %o = fdiv double %t3, %t4
  store double %o, double* %Aj
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; In this test, the counting IV exit value is used, so full strength reduction
; would not reduce register pressure. IndVarSimplify ought to simplify such
; cases away, but it's useful here to verify that LSR's register pressure
; heuristics are working as expected.

; CHECK: count_me_0:
; CHECK: movsd   (%rsi,%rax,8), %xmm0
; CHECK: mulsd   (%rdx,%rax,8), %xmm0
; CHECK: movsd   %xmm0, (%rdi,%rax,8)
; CHECK: incq    %rax
; CHECK: cmpq    %rax, %rcx
; CHECK: jne

define i64 @count_me_0(double* nocapture %A, double* nocapture %B, double* nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %Ai = getelementptr inbounds double* %A, i64 %i
  %Bi = getelementptr inbounds double* %B, i64 %i
  %Ci = getelementptr inbounds double* %C, i64 %i
  %t1 = load double* %Bi
  %t2 = load double* %Ci
  %m = fmul double %t1, %t2
  store double %m, double* %Ai
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  %q = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  ret i64 %q
}

; In this test, the trip count value is used, so full strength reduction
; would not reduce register pressure.
; (though it would reduce register pressure inside the loop...)

; CHECK: count_me_1:
; CHECK: movsd   (%rsi,%rax,8), %xmm0
; CHECK: mulsd   (%rdx,%rax,8), %xmm0
; CHECK: movsd   %xmm0, (%rdi,%rax,8)
; CHECK: incq    %rax
; CHECK: cmpq    %rax, %rcx
; CHECK: jne

define i64 @count_me_1(double* nocapture %A, double* nocapture %B, double* nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %Ai = getelementptr inbounds double* %A, i64 %i
  %Bi = getelementptr inbounds double* %B, i64 %i
  %Ci = getelementptr inbounds double* %C, i64 %i
  %t1 = load double* %Bi
  %t2 = load double* %Ci
  %m = fmul double %t1, %t2
  store double %m, double* %Ai
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  %q = phi i64 [ 0, %entry ], [ %n, %loop ]
  ret i64 %q
}

; Full strength reduction doesn't save any registers here because the
; loop tripcount is a constant.

; CHECK: count_me_2:
; CHECK: movl    $10, %eax
; CHECK: align
; CHECK: BB6_1:
; CHECK: movsd   -40(%rdi,%rax,8), %xmm0
; CHECK: addsd   -40(%rsi,%rax,8), %xmm0
; CHECK: movsd   %xmm0, -40(%rdx,%rax,8)
; CHECK: movsd   (%rdi,%rax,8), %xmm0
; CHECK: subsd   (%rsi,%rax,8), %xmm0
; CHECK: movsd   %xmm0, (%rdx,%rax,8)
; CHECK: incq    %rax
; CHECK: cmpq    $5010, %rax
; CHECK: jne

define void @count_me_2(double* nocapture %A, double* nocapture %B, double* nocapture %C) nounwind {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %i5 = add i64 %i, 5
  %Ai = getelementptr double* %A, i64 %i5
  %t2 = load double* %Ai
  %Bi = getelementptr double* %B, i64 %i5
  %t4 = load double* %Bi
  %t5 = fadd double %t2, %t4
  %Ci = getelementptr double* %C, i64 %i5
  store double %t5, double* %Ci
  %i10 = add i64 %i, 10
  %Ai10 = getelementptr double* %A, i64 %i10
  %t9 = load double* %Ai10
  %Bi10 = getelementptr double* %B, i64 %i10
  %t11 = load double* %Bi10
  %t12 = fsub double %t9, %t11
  %Ci10 = getelementptr double* %C, i64 %i10
  store double %t12, double* %Ci10
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 5000
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; This should be fully strength-reduced to reduce register pressure.

; CHECK: full_me_1:
; CHECK: align
; CHECK: BB7_1:
; CHECK: movsd   (%rdi), %xmm0
; CHECK: addsd   (%rsi), %xmm0
; CHECK: movsd   %xmm0, (%rdx)
; CHECK: movsd   40(%rdi), %xmm0
; CHECK: subsd   40(%rsi), %xmm0
; CHECK: movsd   %xmm0, 40(%rdx)
; CHECK: addq    $8, %rdi
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: decq    %rcx
; CHECK: jne

define void @full_me_1(double* nocapture %A, double* nocapture %B, double* nocapture %C, i64 %n) nounwind {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %i5 = add i64 %i, 5
  %Ai = getelementptr double* %A, i64 %i5
  %t2 = load double* %Ai
  %Bi = getelementptr double* %B, i64 %i5
  %t4 = load double* %Bi
  %t5 = fadd double %t2, %t4
  %Ci = getelementptr double* %C, i64 %i5
  store double %t5, double* %Ci
  %i10 = add i64 %i, 10
  %Ai10 = getelementptr double* %A, i64 %i10
  %t9 = load double* %Ai10
  %Bi10 = getelementptr double* %B, i64 %i10
  %t11 = load double* %Bi10
  %t12 = fsub double %t9, %t11
  %Ci10 = getelementptr double* %C, i64 %i10
  store double %t12, double* %Ci10
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; This is a variation on full_me_0 in which the 0,+,1 induction variable
; has a non-address use, pinning that value in a register.

; CHECK: count_me_3:
; CHECK: call
; CHECK: movsd   (%r15,%r13,8), %xmm0
; CHECK: mulsd   (%r14,%r13,8), %xmm0
; CHECK: movsd   %xmm0, (%r12,%r13,8)
; CHECK: incq    %r13
; CHECK: cmpq    %r13, %rbx
; CHECK: jne

declare void @use(i64)

define void @count_me_3(double* nocapture %A, double* nocapture %B, double* nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  call void @use(i64 %i)
  %Ai = getelementptr inbounds double* %A, i64 %i
  %Bi = getelementptr inbounds double* %B, i64 %i
  %Ci = getelementptr inbounds double* %C, i64 %i
  %t1 = load double* %Bi
  %t2 = load double* %Ci
  %m = fmul double %t1, %t2
  store double %m, double* %Ai
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; LSR should use only one indvar for the inner loop.
; rdar://7657764

; CHECK: asd:
; CHECK: BB9_5:
; CHECK-NEXT: addl  (%r{{[^,]*}},%rdi,4), %e
; CHECK-NEXT: incq  %rdi
; CHECK-NEXT: cmpq  %rdi, %r{{[^,]*}}
; CHECK-NEXT: jg

%struct.anon = type { i32, [4200 x i32] }

@bars = common global [123123 x %struct.anon] zeroinitializer, align 32 ; <[123123 x %struct.anon]*> [#uses=2]

define i32 @asd(i32 %n) nounwind readonly {
entry:
  %0 = icmp sgt i32 %n, 0                         ; <i1> [#uses=1]
  br i1 %0, label %bb.nph14, label %bb5

bb.nph14:                                         ; preds = %entry
  %tmp18 = zext i32 %n to i64                     ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb3, %bb.nph14
  %indvar16 = phi i64 [ 0, %bb.nph14 ], [ %indvar.next17, %bb3 ] ; <i64> [#uses=3]
  %s.113 = phi i32 [ 0, %bb.nph14 ], [ %s.0.lcssa, %bb3 ] ; <i32> [#uses=2]
  %scevgep2526 = getelementptr [123123 x %struct.anon]* @bars, i64 0, i64 %indvar16, i32 0 ; <i32*> [#uses=1]
  %1 = load i32* %scevgep2526, align 4            ; <i32> [#uses=2]
  %2 = icmp sgt i32 %1, 0                         ; <i1> [#uses=1]
  br i1 %2, label %bb.nph, label %bb3

bb.nph:                                           ; preds = %bb
  %tmp23 = sext i32 %1 to i64                     ; <i64> [#uses=1]
  br label %bb1

bb1:                                              ; preds = %bb.nph, %bb1
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp19, %bb1 ] ; <i64> [#uses=2]
  %s.07 = phi i32 [ %s.113, %bb.nph ], [ %4, %bb1 ] ; <i32> [#uses=1]
  %c.08 = getelementptr [123123 x %struct.anon]* @bars, i64 0, i64 %indvar16, i32 1, i64 %indvar ; <i32*> [#uses=1]
  %3 = load i32* %c.08, align 4                   ; <i32> [#uses=1]
  %4 = add nsw i32 %3, %s.07                      ; <i32> [#uses=2]
  %tmp19 = add i64 %indvar, 1                     ; <i64> [#uses=2]
  %5 = icmp sgt i64 %tmp23, %tmp19                ; <i1> [#uses=1]
  br i1 %5, label %bb1, label %bb3

bb3:                                              ; preds = %bb1, %bb
  %s.0.lcssa = phi i32 [ %s.113, %bb ], [ %4, %bb1 ] ; <i32> [#uses=2]
  %indvar.next17 = add i64 %indvar16, 1           ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next17, %tmp18  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb5, label %bb

bb5:                                              ; preds = %bb3, %entry
  %s.1.lcssa = phi i32 [ 0, %entry ], [ %s.0.lcssa, %bb3 ] ; <i32> [#uses=1]
  ret i32 %s.1.lcssa
}
