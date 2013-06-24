; RUN: opt < %s -loop-vectorize -force-vector-width=2 -force-vector-unroll=1 -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-unroll=1 -S | FileCheck %s -check-prefix=WIDTH

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Vectorization with dependence checks.

; No plausible dependence - can be vectorized.
;  for (i = 0; i < 1024; ++i)
;    A[i] = A[i + 1] + 1;

; CHECK: f1_vec
; CHECK: <2 x i32>

define void @f1_vec(i32* %A) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add i32 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32* %A, i32 %indvars.iv.next
  %0 = load i32* %arrayidx, align 4
  %add1 = add nsw i32 %0, 1
  %arrayidx3 = getelementptr inbounds i32* %A, i32 %indvars.iv
  store i32 %add1, i32* %arrayidx3, align 4
  %exitcond = icmp ne i32 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; Plausible dependence of distance 1 - can't be vectorized.
;  for (i = 0; i < 1024; ++i)
;    A[i+1] = A[i] + 1;

; CHECK: f2_novec
; CHECK-NOT: <2 x i32>

define void @f2_novec(i32* %A) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32* %A, i32 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %indvars.iv.next = add i32 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds i32* %A, i32 %indvars.iv.next
  store i32 %add, i32* %arrayidx3, align 4
  %exitcond = icmp ne i32 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; Plausible dependence of distance 2 - can be vectorized with a width of 2.
;  for (i = 0; i < 1024; ++i)
;    A[i+2] = A[i] + 1;

; CHECK: f3_vec_len
; CHECK: <2 x i32>

; WIDTH: f3_vec_len
; WIDTH-NOT: <4 x i32>

define void @f3_vec_len(i32* %A) {
entry:
  br label %for.body

for.body:
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = sext i32 %i.01 to i64
  %arrayidx = getelementptr inbounds i32* %A, i64 %idxprom
  %0 = load i32* %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %add1 = add nsw i32 %i.01, 2
  %idxprom2 = sext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds i32* %A, i64 %idxprom2
  store i32 %add, i32* %arrayidx3, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; Plausible dependence of distance 1 - cannot be vectorized (without reordering
; accesses).
;   for (i = 0; i < 1024; ++i) {
;     B[i] = A[i];
;     A[i] = B[i + 1];
;   }

; CHECK: f5
; CHECK-NOT: <2 x i32>

define void @f5(i32*  %A, i32* %B) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32* %B, i64 %indvars.iv
  store i32 %0, i32* %arrayidx2, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %arrayidx4 = getelementptr inbounds i32* %B, i64 %indvars.iv.next
  %1 = load i32* %arrayidx4, align 4
  store i32 %1, i32* %arrayidx, align 4
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; Dependence through a phi node - must not vectorize.
;   for (i = 0; i < 1024; ++i) {
;     a[i+1] = tmp;
;     tmp = a[i];
;   }

; CHECK: f6
; CHECK-NOT: <2 x i32>

define i32 @f6(i32* %a, i32 %tmp) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp.addr.08 = phi i32 [ %tmp, %entry ], [ %0, %for.body ]
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32* %a, i64 %indvars.iv.next
  store i32 %tmp.addr.08, i32* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32* %a, i64 %indvars.iv
  %0 = load i32* %arrayidx3, align 4
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret i32 undef
}

; Don't vectorize true loop carried dependencies that are not a multiple of the
; vector width.
; Example:
;   for (int i = ...; ++i) {
;     a[i] = a[i-3] + ...;
; It is a bad idea to vectorize this loop because store-load forwarding will not
; happen.
;

; CHECK: @nostoreloadforward
; CHECK-NOT: <2 x i32>

define void @nostoreloadforward(i32* %A) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 16, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = add nsw i64 %indvars.iv, -3
  %arrayidx = getelementptr inbounds i32* %A, i64 %0
  %1 = load i32* %arrayidx, align 4
  %2 = add nsw i64 %indvars.iv, 4
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %2
  %3 = load i32* %arrayidx2, align 4
  %add3 = add nsw i32 %3, %1
  %arrayidx5 = getelementptr inbounds i32* %A, i64 %indvars.iv
  store i32 %add3, i32* %arrayidx5, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; Example:
;   for (int i = ...; ++i) {
;     a[i] = b[i];
;     c[i] = a[i-3] + ...;
; It is a bad idea to vectorize this loop because store-load forwarding will not
; happen.
;

; CHECK: @nostoreloadforward2
; CHECK-NOT: <2 x i32>

define void @nostoreloadforward2(i32* noalias %A, i32* noalias %B, i32* noalias %C) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 16, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32* %B, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  store i32 %0, i32* %arrayidx2, align 4
  %1 = add nsw i64 %indvars.iv, -3
  %arrayidx4 = getelementptr inbounds i32* %A, i64 %1
  %2 = load i32* %arrayidx4, align 4
  %arrayidx6 = getelementptr inbounds i32* %C, i64 %indvars.iv
  store i32 %2, i32* %arrayidx6, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}
