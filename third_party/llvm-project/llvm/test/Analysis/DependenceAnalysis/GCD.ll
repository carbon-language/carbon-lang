; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s -check-prefix=DELIN

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"


;;  for (long int i = 0; i < 100; i++)
;;    for (long int j = 0; j < 100; j++) {
;;      A[2*i - 4*j] = i;
;;      *B++ = A[6*i + 8*j];

define void @gcd0(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; DELIN-LABEL: gcd0
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - flow [=> *|<]!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc8
  %B.addr.04 = phi i32* [ %B, %entry ], [ %scevgep, %for.inc8 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc9, %for.inc8 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i32* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %conv = trunc i64 %i.03 to i32
  %mul = shl nsw i64 %i.03, 1
  %mul4 = shl nsw i64 %j.02, 2
  %sub = sub nsw i64 %mul, %mul4
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %sub
  store i32 %conv, i32* %arrayidx, align 4
  %mul5 = mul nsw i64 %i.03, 6
  %mul6 = shl nsw i64 %j.02, 3
  %add = add nsw i64 %mul5, %mul6
  %arrayidx7 = getelementptr inbounds i32, i32* %A, i64 %add
  %0 = load i32, i32* %arrayidx7, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.11, i64 1
  store i32 %0, i32* %B.addr.11, align 4
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 100
  br i1 %exitcond, label %for.body3, label %for.inc8

for.inc8:                                         ; preds = %for.body3
  %scevgep = getelementptr i32, i32* %B.addr.04, i64 100
  %inc9 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc9, 100
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end10

for.end10:                                        ; preds = %for.inc8
  ret void
}


;;  for (long int i = 0; i < 100; i++)
;;    for (long int j = 0; j < 100; j++) {
;;      A[2*i - 4*j] = i;
;;      *B++ = A[6*i + 8*j + 1];

define void @gcd1(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; DELIN-LABEL: gcd1
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc9
  %B.addr.04 = phi i32* [ %B, %entry ], [ %scevgep, %for.inc9 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc10, %for.inc9 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i32* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %conv = trunc i64 %i.03 to i32
  %mul = shl nsw i64 %i.03, 1
  %mul4 = shl nsw i64 %j.02, 2
  %sub = sub nsw i64 %mul, %mul4
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %sub
  store i32 %conv, i32* %arrayidx, align 4
  %mul5 = mul nsw i64 %i.03, 6
  %mul6 = shl nsw i64 %j.02, 3
  %add = add nsw i64 %mul5, %mul6
  %add7 = or i64 %add, 1
  %arrayidx8 = getelementptr inbounds i32, i32* %A, i64 %add7
  %0 = load i32, i32* %arrayidx8, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.11, i64 1
  store i32 %0, i32* %B.addr.11, align 4
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 100
  br i1 %exitcond, label %for.body3, label %for.inc9

for.inc9:                                         ; preds = %for.body3
  %scevgep = getelementptr i32, i32* %B.addr.04, i64 100
  %inc10 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc10, 100
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end11

for.end11:                                        ; preds = %for.inc9
  ret void
}


;;  for (long int i = 0; i < 100; i++)
;;    for (long int j = 0; j < 100; j++) {
;;      A[2*i - 4*j + 1] = i;
;;      *B++ = A[6*i + 8*j];

define void @gcd2(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; DELIN-LABEL: gcd2
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc9
  %B.addr.04 = phi i32* [ %B, %entry ], [ %scevgep, %for.inc9 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc10, %for.inc9 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i32* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %conv = trunc i64 %i.03 to i32
  %mul = shl nsw i64 %i.03, 1
  %mul4 = shl nsw i64 %j.02, 2
  %sub = sub nsw i64 %mul, %mul4
  %add5 = or i64 %sub, 1
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %add5
  store i32 %conv, i32* %arrayidx, align 4
  %mul5 = mul nsw i64 %i.03, 6
  %mul6 = shl nsw i64 %j.02, 3
  %add7 = add nsw i64 %mul5, %mul6
  %arrayidx8 = getelementptr inbounds i32, i32* %A, i64 %add7
  %0 = load i32, i32* %arrayidx8, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.11, i64 1
  store i32 %0, i32* %B.addr.11, align 4
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 100
  br i1 %exitcond, label %for.body3, label %for.inc9

for.inc9:                                         ; preds = %for.body3
  %scevgep = getelementptr i32, i32* %B.addr.04, i64 100
  %inc10 = add nsw i64 %i.03, 1
  %exitcond6 = icmp ne i64 %inc10, 100
  br i1 %exitcond6, label %for.cond1.preheader, label %for.end11

for.end11:                                        ; preds = %for.inc9
  ret void
}


;;  for (long int i = 0; i < 100; i++)
;;    for (long int j = 0; j < 100; j++) {
;;      A[i + 2*j] = i;
;;      *B++ = A[i + 2*j - 1];

define void @gcd3(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; DELIN-LABEL: gcd3
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - flow [<> *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc7
  %B.addr.04 = phi i32* [ %B, %entry ], [ %scevgep, %for.inc7 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc8, %for.inc7 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i32* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %conv = trunc i64 %i.03 to i32
  %mul = shl nsw i64 %j.02, 1
  %add = add nsw i64 %i.03, %mul
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %add
  store i32 %conv, i32* %arrayidx, align 4
  %mul4 = shl nsw i64 %j.02, 1
  %add5 = add nsw i64 %i.03, %mul4
  %sub = add nsw i64 %add5, -1
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i64 %sub
  %0 = load i32, i32* %arrayidx6, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.11, i64 1
  store i32 %0, i32* %B.addr.11, align 4
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 100
  br i1 %exitcond, label %for.body3, label %for.inc7

for.inc7:                                         ; preds = %for.body3
  %scevgep = getelementptr i32, i32* %B.addr.04, i64 100
  %inc8 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc8, 100
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end9

for.end9:                                         ; preds = %for.inc7
  ret void
}


;;  for (long int i = 0; i < 100; i++)
;;    for (long int j = 0; j < 100; j++) {
;;      A[5*i + 10*j*M + 9*M*N] = i;
;;      *B++ = A[15*i + 20*j*M - 21*N*M + 4];

define void @gcd4(i32* %A, i32* %B, i64 %M, i64 %N) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; DELIN-LABEL: gcd4
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc17
  %B.addr.04 = phi i32* [ %B, %entry ], [ %scevgep, %for.inc17 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc18, %for.inc17 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i32* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, 5
  %mul4 = mul nsw i64 %j.02, 10
  %mul5 = mul nsw i64 %mul4, %M
  %add = add nsw i64 %mul, %mul5
  %mul6 = mul nsw i64 %M, 9
  %mul7 = mul nsw i64 %mul6, %N
  %add8 = add nsw i64 %add, %mul7
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %add8
  store i32 %conv, i32* %arrayidx, align 4
  %mul9 = mul nsw i64 %i.03, 15
  %mul10 = mul nsw i64 %j.02, 20
  %mul11 = mul nsw i64 %mul10, %M
  %add12 = add nsw i64 %mul9, %mul11
  %mul13 = mul nsw i64 %N, 21
  %mul14 = mul nsw i64 %mul13, %M
  %sub = sub nsw i64 %add12, %mul14
  %add15 = add nsw i64 %sub, 4
  %arrayidx16 = getelementptr inbounds i32, i32* %A, i64 %add15
  %0 = load i32, i32* %arrayidx16, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.11, i64 1
  store i32 %0, i32* %B.addr.11, align 4
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 100
  br i1 %exitcond, label %for.body3, label %for.inc17

for.inc17:                                        ; preds = %for.body3
  %scevgep = getelementptr i32, i32* %B.addr.04, i64 100
  %inc18 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc18, 100
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end19

for.end19:                                        ; preds = %for.inc17
  ret void
}


;;  for (long int i = 0; i < 100; i++)
;;    for (long int j = 0; j < 100; j++) {
;;      A[5*i + 10*j*M + 9*M*N] = i;
;;      *B++ = A[15*i + 20*j*M - 21*N*M + 5];

define void @gcd5(i32* %A, i32* %B, i64 %M, i64 %N) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; DELIN-LABEL: gcd5
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - flow [<> *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc17
  %B.addr.04 = phi i32* [ %B, %entry ], [ %scevgep, %for.inc17 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc18, %for.inc17 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i32* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, 5
  %mul4 = mul nsw i64 %j.02, 10
  %mul5 = mul nsw i64 %mul4, %M
  %add = add nsw i64 %mul, %mul5
  %mul6 = mul nsw i64 %M, 9
  %mul7 = mul nsw i64 %mul6, %N
  %add8 = add nsw i64 %add, %mul7
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %add8
  store i32 %conv, i32* %arrayidx, align 4
  %mul9 = mul nsw i64 %i.03, 15
  %mul10 = mul nsw i64 %j.02, 20
  %mul11 = mul nsw i64 %mul10, %M
  %add12 = add nsw i64 %mul9, %mul11
  %mul13 = mul nsw i64 %N, 21
  %mul14 = mul nsw i64 %mul13, %M
  %sub = sub nsw i64 %add12, %mul14
  %add15 = add nsw i64 %sub, 5
  %arrayidx16 = getelementptr inbounds i32, i32* %A, i64 %add15
  %0 = load i32, i32* %arrayidx16, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.11, i64 1
  store i32 %0, i32* %B.addr.11, align 4
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 100
  br i1 %exitcond, label %for.body3, label %for.inc17

for.inc17:                                        ; preds = %for.body3
  %scevgep = getelementptr i32, i32* %B.addr.04, i64 100
  %inc18 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc18, 100
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end19

for.end19:                                        ; preds = %for.inc17
  ret void
}


;;  for (long int i = 0; i < n; i++)
;;    for (long int j = 0; j < n; j++) {
;;      A[2*i][4*j] = i;
;;      *B++ = A[8*i][6*j + 1];

define void @gcd6(i64 %n, i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  %cmp4 = icmp sgt i64 %n, 0
  br i1 %cmp4, label %for.cond1.preheader.preheader, label %for.end12

; DELIN-LABEL: gcd6
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - output [* *]!

for.cond1.preheader.preheader:                    ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc10
  %i.06 = phi i64 [ %inc11, %for.inc10 ], [ 0, %for.cond1.preheader.preheader ]
  %B.addr.05 = phi i32* [ %B.addr.1.lcssa, %for.inc10 ], [ %B, %for.cond1.preheader.preheader ]
  %cmp21 = icmp sgt i64 %n, 0
  br i1 %cmp21, label %for.body3.preheader, label %for.inc10

for.body3.preheader:                              ; preds = %for.cond1.preheader
  br label %for.body3

for.body3:                                        ; preds = %for.body3.preheader, %for.body3
  %j.03 = phi i64 [ %inc, %for.body3 ], [ 0, %for.body3.preheader ]
  %B.addr.12 = phi i32* [ %incdec.ptr, %for.body3 ], [ %B.addr.05, %for.body3.preheader ]
  %conv = trunc i64 %i.06 to i32
  %mul = shl nsw i64 %j.03, 2
  %mul4 = shl nsw i64 %i.06, 1
  %0 = mul nsw i64 %mul4, %n
  %arrayidx.sum = add i64 %0, %mul
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %arrayidx.sum
  store i32 %conv, i32* %arrayidx5, align 4
  %mul6 = mul nsw i64 %j.03, 6
  %add7 = or i64 %mul6, 1
  %mul7 = shl nsw i64 %i.06, 3
  %1 = mul nsw i64 %mul7, %n
  %arrayidx8.sum = add i64 %1, %add7
  %arrayidx9 = getelementptr inbounds i32, i32* %A, i64 %arrayidx8.sum
  %2 = load i32, i32* %arrayidx9, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.12, i64 1
  store i32 %2, i32* %B.addr.12, align 4
  %inc = add nsw i64 %j.03, 1
  %exitcond = icmp ne i64 %inc, %n
  br i1 %exitcond, label %for.body3, label %for.inc10.loopexit

for.inc10.loopexit:                               ; preds = %for.body3
  %scevgep = getelementptr i32, i32* %B.addr.05, i64 %n
  br label %for.inc10

for.inc10:                                        ; preds = %for.inc10.loopexit, %for.cond1.preheader
  %B.addr.1.lcssa = phi i32* [ %B.addr.05, %for.cond1.preheader ], [ %scevgep, %for.inc10.loopexit ]
  %inc11 = add nsw i64 %i.06, 1
  %exitcond8 = icmp ne i64 %inc11, %n
  br i1 %exitcond8, label %for.cond1.preheader, label %for.end12.loopexit

for.end12.loopexit:                               ; preds = %for.inc10
  br label %for.end12

for.end12:                                        ; preds = %for.end12.loopexit, %entry
  ret void
}


;;  for (int i = 0; i < n; i++)
;;   for (int j = 0; j < n; j++) {
;;    A[2*i][4*j] = i;
;;   *B++ = A[8*i][6*j + 1];

define void @gcd7(i32 %n, i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  %0 = zext i32 %n to i64
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.cond1.preheader.preheader, label %for.end15

; DELIN-LABEL: gcd7
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - flow [* *|<]!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - output [* *]!

for.cond1.preheader.preheader:                    ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc13
  %indvars.iv8 = phi i64 [ 0, %for.cond1.preheader.preheader ], [ %indvars.iv.next9, %for.inc13 ]
  %B.addr.05 = phi i32* [ %B.addr.1.lcssa, %for.inc13 ], [ %B, %for.cond1.preheader.preheader ]
  %1 = add i32 %n, -1
  %2 = zext i32 %1 to i64
  %3 = add i64 %2, 1
  %cmp21 = icmp sgt i32 %n, 0
  br i1 %cmp21, label %for.body3.preheader, label %for.inc13

for.body3.preheader:                              ; preds = %for.cond1.preheader
  br label %for.body3

for.body3:                                        ; preds = %for.body3.preheader, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body3.preheader ], [ %indvars.iv.next, %for.body3 ]
  %B.addr.12 = phi i32* [ %incdec.ptr, %for.body3 ], [ %B.addr.05, %for.body3.preheader ]
  %4 = trunc i64 %indvars.iv to i32
  %mul = shl nsw i32 %4, 2
  %idxprom = sext i32 %mul to i64
  %5 = trunc i64 %indvars.iv8 to i32
  %mul4 = shl nsw i32 %5, 1
  %idxprom5 = sext i32 %mul4 to i64
  %6 = mul nsw i64 %idxprom5, %0
  %arrayidx.sum = add i64 %6, %idxprom
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i64 %arrayidx.sum
  %7 = trunc i64 %indvars.iv8 to i32
  store i32 %7, i32* %arrayidx6, align 4
  %8 = trunc i64 %indvars.iv to i32
  %mul7 = mul nsw i32 %8, 6
  %add7 = or i32 %mul7, 1
  %idxprom8 = sext i32 %add7 to i64
  %9 = trunc i64 %indvars.iv8 to i32
  %mul9 = shl nsw i32 %9, 3
  %idxprom10 = sext i32 %mul9 to i64
  %10 = mul nsw i64 %idxprom10, %0
  %arrayidx11.sum = add i64 %10, %idxprom8
  %arrayidx12 = getelementptr inbounds i32, i32* %A, i64 %arrayidx11.sum
  %11 = load i32, i32* %arrayidx12, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.12, i64 1
  store i32 %11, i32* %B.addr.12, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.body3, label %for.inc13.loopexit

for.inc13.loopexit:                               ; preds = %for.body3
  %scevgep = getelementptr i32, i32* %B.addr.05, i64 %3
  br label %for.inc13

for.inc13:                                        ; preds = %for.inc13.loopexit, %for.cond1.preheader
  %B.addr.1.lcssa = phi i32* [ %B.addr.05, %for.cond1.preheader ], [ %scevgep, %for.inc13.loopexit ]
  %indvars.iv.next9 = add i64 %indvars.iv8, 1
  %lftr.wideiv10 = trunc i64 %indvars.iv.next9 to i32
  %exitcond11 = icmp ne i32 %lftr.wideiv10, %n
  br i1 %exitcond11, label %for.cond1.preheader, label %for.end15.loopexit

for.end15.loopexit:                               ; preds = %for.inc13
  br label %for.end15

for.end15:                                        ; preds = %for.end15.loopexit, %entry
  ret void
}


;;  for (int i = 0; i < n; i++)
;;    for (int j = 0; j < n; j++) {
;;      A[n*2*i + 4*j] = i;
;;      *B++ = A[n*8*i + 6*j + 1];

define void @gcd8(i32 %n, i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.cond1.preheader.preheader, label %for.end15

; DELIN-LABEL: gcd8
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - flow [* *|<]!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - output [* *]!

for.cond1.preheader.preheader:                    ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc13
  %i.06 = phi i32 [ %inc14, %for.inc13 ], [ 0, %for.cond1.preheader.preheader ]
  %B.addr.05 = phi i32* [ %B.addr.1.lcssa, %for.inc13 ], [ %B, %for.cond1.preheader.preheader ]
  %0 = add i32 %n, -1
  %1 = zext i32 %0 to i64
  %2 = add i64 %1, 1
  %cmp21 = icmp sgt i32 %n, 0
  br i1 %cmp21, label %for.body3.preheader, label %for.inc13

for.body3.preheader:                              ; preds = %for.cond1.preheader
  br label %for.body3

for.body3:                                        ; preds = %for.body3.preheader, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body3.preheader ], [ %indvars.iv.next, %for.body3 ]
  %B.addr.12 = phi i32* [ %incdec.ptr, %for.body3 ], [ %B.addr.05, %for.body3.preheader ]
  %mul = shl nsw i32 %n, 1
  %mul4 = mul nsw i32 %mul, %i.06
  %3 = trunc i64 %indvars.iv to i32
  %mul5 = shl nsw i32 %3, 2
  %add = add nsw i32 %mul4, %mul5
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  store i32 %i.06, i32* %arrayidx, align 4
  %mul6 = shl nsw i32 %n, 3
  %mul7 = mul nsw i32 %mul6, %i.06
  %4 = trunc i64 %indvars.iv to i32
  %mul8 = mul nsw i32 %4, 6
  %add9 = add nsw i32 %mul7, %mul8
  %add10 = or i32 %add9, 1
  %idxprom11 = sext i32 %add10 to i64
  %arrayidx12 = getelementptr inbounds i32, i32* %A, i64 %idxprom11
  %5 = load i32, i32* %arrayidx12, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.12, i64 1
  store i32 %5, i32* %B.addr.12, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.body3, label %for.inc13.loopexit

for.inc13.loopexit:                               ; preds = %for.body3
  %scevgep = getelementptr i32, i32* %B.addr.05, i64 %2
  br label %for.inc13

for.inc13:                                        ; preds = %for.inc13.loopexit, %for.cond1.preheader
  %B.addr.1.lcssa = phi i32* [ %B.addr.05, %for.cond1.preheader ], [ %scevgep, %for.inc13.loopexit ]
  %inc14 = add nsw i32 %i.06, 1
  %exitcond7 = icmp ne i32 %inc14, %n
  br i1 %exitcond7, label %for.cond1.preheader, label %for.end15.loopexit

for.end15.loopexit:                               ; preds = %for.inc13
  br label %for.end15

for.end15:                                        ; preds = %for.end15.loopexit, %entry
  ret void
}


;;  for (unsigned i = 0; i < n; i++)
;;    for (unsigned j = 0; j < n; j++) {
;;      A[2*i][4*j] = i;
;;      *B++ = A[8*i][6*j + 1];

define void @gcd9(i32 %n, i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  %0 = zext i32 %n to i64
  %cmp4 = icmp eq i32 %n, 0
  br i1 %cmp4, label %for.end15, label %for.cond1.preheader.preheader

; DELIN-LABEL: gcd9
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - flow [* *|<]!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - output [* *]!

for.cond1.preheader.preheader:                    ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc13
  %indvars.iv8 = phi i64 [ 0, %for.cond1.preheader.preheader ], [ %indvars.iv.next9, %for.inc13 ]
  %B.addr.05 = phi i32* [ %B.addr.1.lcssa, %for.inc13 ], [ %B, %for.cond1.preheader.preheader ]
  %1 = add i32 %n, -1
  %2 = zext i32 %1 to i64
  %3 = add i64 %2, 1
  %cmp21 = icmp eq i32 %n, 0
  br i1 %cmp21, label %for.inc13, label %for.body3.preheader

for.body3.preheader:                              ; preds = %for.cond1.preheader
  br label %for.body3

for.body3:                                        ; preds = %for.body3.preheader, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body3.preheader ], [ %indvars.iv.next, %for.body3 ]
  %B.addr.12 = phi i32* [ %incdec.ptr, %for.body3 ], [ %B.addr.05, %for.body3.preheader ]
  %4 = trunc i64 %indvars.iv to i32
  %mul = shl i32 %4, 2
  %idxprom = zext i32 %mul to i64
  %5 = trunc i64 %indvars.iv8 to i32
  %mul4 = shl i32 %5, 1
  %idxprom5 = zext i32 %mul4 to i64
  %6 = mul nsw i64 %idxprom5, %0
  %arrayidx.sum = add i64 %6, %idxprom
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i64 %arrayidx.sum
  %7 = trunc i64 %indvars.iv8 to i32
  store i32 %7, i32* %arrayidx6, align 4
  %8 = trunc i64 %indvars.iv to i32
  %mul7 = mul i32 %8, 6
  %add7 = or i32 %mul7, 1
  %idxprom8 = zext i32 %add7 to i64
  %9 = trunc i64 %indvars.iv8 to i32
  %mul9 = shl i32 %9, 3
  %idxprom10 = zext i32 %mul9 to i64
  %10 = mul nsw i64 %idxprom10, %0
  %arrayidx11.sum = add i64 %10, %idxprom8
  %arrayidx12 = getelementptr inbounds i32, i32* %A, i64 %arrayidx11.sum
  %11 = load i32, i32* %arrayidx12, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.12, i64 1
  store i32 %11, i32* %B.addr.12, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.body3, label %for.inc13.loopexit

for.inc13.loopexit:                               ; preds = %for.body3
  %scevgep = getelementptr i32, i32* %B.addr.05, i64 %3
  br label %for.inc13

for.inc13:                                        ; preds = %for.inc13.loopexit, %for.cond1.preheader
  %B.addr.1.lcssa = phi i32* [ %B.addr.05, %for.cond1.preheader ], [ %scevgep, %for.inc13.loopexit ]
  %indvars.iv.next9 = add i64 %indvars.iv8, 1
  %lftr.wideiv10 = trunc i64 %indvars.iv.next9 to i32
  %exitcond11 = icmp ne i32 %lftr.wideiv10, %n
  br i1 %exitcond11, label %for.cond1.preheader, label %for.end15.loopexit

for.end15.loopexit:                               ; preds = %for.inc13
  br label %for.end15

for.end15:                                        ; preds = %for.end15.loopexit, %entry
  ret void
}
