; RUN: llc < %s -march=x86-64 | FileCheck %s
target datalayout = "e-p:64:64:64"
target triple = "x86_64-unknown-unknown"

; Full strength reduction reduces register pressure from 5 to 4 here.

; CHECK: full_me:
; CHECK: movsd   (%rsi), %xmm0
; CHECK: mulsd   (%rdx), %xmm0
; CHECK: movsd   %xmm0, (%rdi)
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: addq    $8, %rdi
; CHECK: decq    %rcx
; CHECK: jne

define void @full_me(double* nocapture %A, double* nocapture %B, double* nocapture %C, i64 %n) nounwind {
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

; This should be fully strength-reduced to reduce register pressure, however
; the current heuristics get distracted by all the reuse with the stride-1
; induction variable first.

; But even so, be clever and start the stride-1 variable at a non-zero value
; to eliminate an in-loop immediate value.

; CHECK: count_me_2:
; CHECK: movl    $5, %eax
; CHECK: align
; CHECK: BB4_1:
; CHECK: movsd   (%rdi,%rax,8), %xmm0
; CHECK: addsd   (%rsi,%rax,8), %xmm0
; CHECK: movsd   %xmm0, (%rdx,%rax,8)
; CHECK: movsd   40(%rdi,%rax,8), %xmm0
; CHECK: addsd   40(%rsi,%rax,8), %xmm0
; CHECK: movsd   %xmm0, 40(%rdx,%rax,8)
; CHECK: incq    %rax
; CHECK: cmpq    $5005, %rax
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
  %t12 = fadd double %t9, %t11
  %Ci10 = getelementptr double* %C, i64 %i10
  store double %t12, double* %Ci10
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 5000
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}
