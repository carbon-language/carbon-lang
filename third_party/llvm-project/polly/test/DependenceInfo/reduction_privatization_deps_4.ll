; RUN: opt %loadPolly -polly-dependences -analyze < %s | FileCheck %s
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     { Stmt_S1[i0] -> Stmt_S2[i0, i0] : 0 <= i0 <= 98; Stmt_S2[i0, i0] -> Stmt_S3[i0] : 0 <= i0 <= 98; Stmt_S3[i0] -> Stmt_S2[o0, i0] : i0 >= 0 and i0 < o0 <= 98; Stmt_S2[i0, i1] -> Stmt_S1[i1] : i0 >= 0 and i0 < i1 <= 98 }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     { Stmt_S1[i0] -> Stmt_S2[i0, i0] : 0 <= i0 <= 98; Stmt_S2[i0, i0] -> Stmt_S3[i0] : 0 <= i0 <= 98; Stmt_S3[i0] -> Stmt_S2[o0, i0] : i0 >= 0 and i0 < o0 <= 98; Stmt_S2[i0, i1] -> Stmt_S1[i1] : i0 >= 0 and i0 < i1 <= 98 }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     { Stmt_S1[i0] -> Stmt_S2[i0, i0] : 0 <= i0 <= 98; Stmt_S2[i0, i0] -> Stmt_S3[i0] : 0 <= i0 <= 98; Stmt_S3[i0] -> Stmt_S2[o0, i0] : i0 >= 0 and i0 < o0 <= 98; Stmt_S2[i0, i1] -> Stmt_S1[i1] : i0 >= 0 and i0 < i1 <= 98 }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     { Stmt_S2[i0, i1] -> Stmt_S2[1 + i0, i1] : (i0 >= 0 and 2 + i0 <= i1 <= 99) or (i0 <= 97 and 0 <= i1 < i0) }
;
;    void f(int *sum) {
;      for (int i = 0; i < 99; i++) {
; S1:    sum[i] += 42;
;        for (int j = 0; j < 100; j++)
; S2:      sum[j] += i * j;
; S3:    sum[i] += 7;
;      }
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* %sum)  {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc8, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc9, %for.inc8 ]
  %exitcond1 = icmp ne i32 %i.0, 99
  br i1 %exitcond1, label %for.body, label %for.end10

for.body:                                         ; preds = %for.cond
  br label %S1

S1:                                               ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %sum, i32 %i.0
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %tmp, 42
  store i32 %add, i32* %arrayidx, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %S1
  %j.0 = phi i32 [ 0, %S1 ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 100
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  br label %S2

S2:                                               ; preds = %for.body3
  %mul = mul nsw i32 %i.0, %j.0
  %arrayidx4 = getelementptr inbounds i32, i32* %sum, i32 %j.0
  %tmp2 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %tmp2, %mul
  store i32 %add5, i32* %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %S2
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %S3

S3:                                               ; preds = %for.end
  %arrayidx6 = getelementptr inbounds i32, i32* %sum, i32 %i.0
  %tmp3 = load i32, i32* %arrayidx6, align 4
  %add7 = add nsw i32 %tmp3, 7
  store i32 %add7, i32* %arrayidx6, align 4
  br label %for.inc8

for.inc8:                                         ; preds = %S3
  %inc9 = add nsw i32 %i.0, 1
  br label %for.cond

for.end10:                                        ; preds = %for.cond
  ret void
}
