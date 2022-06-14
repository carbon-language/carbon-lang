; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 | FileCheck %s

;; void test1(long n, double *A) {
;;     long  i;
;;     for (i = 0; i*n <= n*n; ++i) {
;;         A[i] = i;
;;     }
;;     A[i] = i;
;; }

; CHECK-LABEL: 'Dependence Analysis' for function 'test1':
; CHECK: Src:  store double %conv, ptr %arrayidx, align 8 --> Dst:  store double %conv, ptr %arrayidx, align 8
; CHECK-NEXT:    da analyze - none!
; CHECK: Src:  store double %conv, ptr %arrayidx, align 8 --> Dst:  store double %conv2, ptr %arrayidx3, align 8
; CHECK-NEXT:    da analyze - output [|<]!
; CHECK: Src:  store double %conv2, ptr %arrayidx3, align 8 --> Dst:  store double %conv2, ptr %arrayidx3, align 8
; CHECK-NEXT:    da analyze - none!

define void @test1(i64 noundef %n, ptr nocapture noundef writeonly %A) {
entry:
  %mul1 = mul nsw i64 %n, %n
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.012 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = sitofp i64 %i.012 to double
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %i.012
  store double %conv, ptr %arrayidx, align 8
  %inc = add nuw nsw i64 %i.012, 1
  %mul = mul nsw i64 %inc, %n
  %cmp.not = icmp sgt i64 %mul, %mul1
  br i1 %cmp.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %conv2 = sitofp i64 %inc to double
  %arrayidx3 = getelementptr inbounds double, ptr %A, i64 %inc
  store double %conv2, ptr %arrayidx3, align 8
  ret void
}


;; int test2(unsigned n, float A[][n+1], float B[n+1]) {
;;  for (int i = 0; i <= n; i++) {
;;    long j = 0;
;;    for (; j <= n; ++j) {
;;      B[j] = j;
;;    }
;;    A[i][j] = 123;
;;    for (int k = 0; k <= n; k++) {
;;      A[i][k] = k;
;;    }
;;  }
;;
;; Make sure we can detect depnendence between A[i][j] and A[i][k] conservatively and without crashing.

; CHECK-LABEL: 'Dependence Analysis' for function 'test2':
; CHECK:       Src:  store float 1.230000e+02, ptr %arrayidx7, align 4 --> Dst:  store float %conv13, ptr %arrayidx17, align 4
; CHECK-NEXT:    da analyze - output [*|<]!

define dso_local void @test2(i32 noundef zeroext %n, ptr noundef %A, ptr noalias noundef %B) #0 {
entry:
  %add = add i32 %n, 1
  %0 = zext i32 %add to i64
  %1 = zext i32 %n to i64
  %2 = add nuw nsw i64 %1, 1
  %wide.trip.count9 = zext i32 %add to i64
  br label %for.i

for.i:                                         ; preds = %entry, %for.inc21
  %indvars.iv6 = phi i64 [ 0, %entry ], [ %indvars.iv.next7, %for.inc21 ]
  br label %for.j

for.j:                                        ; preds = %for.i, %for.j
  %j.01 = phi i64 [ 0, %for.i ], [ %inc, %for.j ]
  %conv5 = trunc i64 %j.01 to i32
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %j.01
  store i32 %conv5, ptr %arrayidx, align 4
  %inc = add nuw nsw i64 %j.01, 1
  %exitcond = icmp ne i64 %inc, %2
  br i1 %exitcond, label %for.j, label %for.end

for.end:                                          ; preds = %for.j
  %inc.lcssa = phi i64 [ %inc, %for.j ]
  %3 = mul nuw nsw i64 %indvars.iv6, %0
  %arrayidx6 = getelementptr inbounds float, ptr %A, i64 %3
  %arrayidx7 = getelementptr inbounds float, ptr %arrayidx6, i64 %inc.lcssa
  store float 1.230000e+02, ptr %arrayidx7, align 4
  %wide.trip.count = zext i32 %add to i64
  br label %for.k

for.k:                                       ; preds = %for.end, %for.k
  %indvars.iv = phi i64 [ 0, %for.end ], [ %indvars.iv.next, %for.k ]
  %4 = trunc i64 %indvars.iv to i32
  %conv13 = sitofp i32 %4 to float
  %5 = mul nuw nsw i64 %indvars.iv6, %0
  %arrayidx15 = getelementptr inbounds float, ptr %A, i64 %5
  %arrayidx17 = getelementptr inbounds float, ptr %arrayidx15, i64 %indvars.iv
  store float %conv13, ptr %arrayidx17, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond5 = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond5, label %for.k, label %for.inc21

for.inc21:                                        ; preds = %for.k
  %indvars.iv.next7 = add nuw nsw i64 %indvars.iv6, 1
  %exitcond10 = icmp ne i64 %indvars.iv.next7, %wide.trip.count9
  br i1 %exitcond10, label %for.i, label %for.end23

for.end23:                                        ; preds = %for.inc21
  ret void
}


;; void test3(int n, double *restrict A, double *restrict B) {
;;   for (int i = 0; i < n; ++i) {
;;     int s = 0;
;;     for (; s * s < n * n; ++s) {
;;     }
;;     for (int k = 0; k < n; ++k)
;;       A[s] = 0; // Invariant in innermost loop
;;
;;     A[i] = 1;
;;   }
;; }
;;
;; Make sure we can detect depnendence between A[i] and A[s] conservatively and without crashing.

; CHECK-LABEL: 'Dependence Analysis' for function 'test3':
; CHECK:       Src:  store double 0.000000e+00, ptr %arrayidx, align 8 --> Dst:  store double 1.000000e+00, ptr %arrayidx21, align 8
; CHECK-NEXT:    da analyze - output [*|<]!

define void @test3(i32 noundef %n, ptr noalias noundef %A, ptr noalias noundef %B) {
entry:
  br label %for.i

for.i:                                         ; preds = %for.end19, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc23, %for.end19 ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.s, label %for.end24

for.s:                                        ; preds = %for.i, %for.inc
  %s.0 = phi i32 [ %inc, %for.inc ], [ 0, %for.i ]
  %mul = mul nsw i32 %s.0, %s.0
  %mul2 = mul nsw i32 %n, %n
  %cmp3 = icmp slt i32 %mul, %mul2
  br i1 %cmp3, label %for.inc, label %for.k

for.inc:                                          ; preds = %for.s
  %inc = add nsw i32 %s.0, 1
  br label %for.s

for.k:                                        ; preds = %for.s, %for.body.k
  %k.0 = phi i32 [ %inc10, %for.body.k ], [ 0, %for.s ]
  %cmp6 = icmp slt i32 %k.0, %n
  br i1 %cmp6, label %for.body.k, label %for.end19

for.body.k:                                        ; preds = %for.k
  %idxprom = sext i32 %s.0 to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  store double 0.000000e+00, ptr %arrayidx, align 8
  %inc10 = add nsw i32 %k.0, 1
  br label %for.k

for.end19:                                        ; preds = %for.k
  %idxprom20 = sext i32 %i.0 to i64
  %arrayidx21 = getelementptr inbounds double, ptr %A, i64 %idxprom20
  store double 1.000000e+00, ptr %arrayidx21, align 8
  %inc23 = add nsw i32 %i.0, 1
  br label %for.i

for.end24:                                        ; preds = %for.i
  ret void
}
