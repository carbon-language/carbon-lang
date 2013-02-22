; RUN: opt < %s -O3 -loop-vectorize -force-vector-unroll=1 -force-vector-width=4 -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"

%struct.anon = type { [100 x i32], i32, [100 x i32] }

@Foo = common global %struct.anon zeroinitializer, align 4
@PB = external global i32*
@PA = external global i32*

; int noAlias01 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i] = Foo.B[i] + a;
;   return Foo.A[a];
; }
; CHECK: define i32 @noAlias01
; CHECK: add nsw <4 x i32>
; CHECK ret

define i32 @noAlias01(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32* %i, align 4
  %arrayidx = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %1
  %2 = load i32* %arrayidx, align 4
  %3 = load i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32* %i, align 4
  %arrayidx1 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %4
  store i32 %add, i32* %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32* %a.addr, align 4
  %arrayidx2 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32* %arrayidx2, align 4
  ret i32 %7
}

; int noAlias02 (int a) {
;   int i;
;   for (i=0; i<SIZE-10; i++)
;     Foo.A[i] = Foo.B[i+10] + a;
;   return Foo.A[a];
; }
; CHECK: define i32 @noAlias02
; CHECK: add nsw <4 x i32>
; CHECK ret

define i32 @noAlias02(i32 %a) {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 90
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32* %i, align 4
  %add = add nsw i32 %1, 10
  %arrayidx = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %add
  %2 = load i32* %arrayidx, align 4
  %3 = load i32* %a.addr, align 4
  %add1 = add nsw i32 %2, %3
  %4 = load i32* %i, align 4
  %arrayidx2 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %4
  store i32 %add1, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32* %arrayidx3, align 4
  ret i32 %7
}

; int noAlias03 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i+10] = Foo.B[i] + a;
;   return Foo.A[a];
; }
; CHECK: define i32 @noAlias03
; CHECK: add nsw <4 x i32>
; CHECK ret

define i32 @noAlias03(i32 %a) {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32* %i, align 4
  %arrayidx = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %1
  %2 = load i32* %arrayidx, align 4
  %3 = load i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32* %i, align 4
  %add1 = add nsw i32 %4, 10
  %arrayidx2 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %add1
  store i32 %add, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32* %arrayidx3, align 4
  ret i32 %7
}

; int mayAlias01 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i] = Foo.B[SIZE-i-1] + a;
;   return Foo.A[a];
; }
; CHECK: define i32 @mayAlias01
; CHECK-NOT: add nsw <4 x i32>
; CHECK ret

define i32 @mayAlias01(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 1
  %arrayidx = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32* %arrayidx, align 4
  %3 = load i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32* %i, align 4
  %arrayidx2 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %4
  store i32 %add, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32* %arrayidx3, align 4
  ret i32 %7
}

; int mayAlias04 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     *(PA+i) = *(PB+i) + a;
;   return Foo.A[a];
; }
; CHECK: define i32 @mayAlias04
; CHECK-NOT: add nsw <4 x i32>
; CHECK ret

define i32 @mayAlias04(i32 %a) {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32** @PB, align 4
  %2 = load i32* %i, align 4
  %add.ptr = getelementptr inbounds i32* %1, i32 %2
  %3 = load i32* %add.ptr, align 4
  %4 = load i32* %a.addr, align 4
  %add = add nsw i32 %3, %4
  %5 = load i32** @PA, align 4
  %6 = load i32* %i, align 4
  %add.ptr1 = getelementptr inbounds i32* %5, i32 %6
  store i32 %add, i32* %add.ptr1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i32** @PA, align 4
  %9 = load i32* %a.addr, align 4
  %add.ptr2 = getelementptr inbounds i32* %8, i32 %9
  %10 = load i32* %add.ptr2, align 4
  ret i32 %10
}

; int mayAlias02 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[SIZE-i-1] = Foo.B[i] + a;
;   return Foo.A[a];
; }
; CHECK: define i32 @mayAlias02
; CHECK-NOT: add nsw <4 x i32>
; CHECK ret

define i32 @mayAlias02(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32* %i, align 4
  %arrayidx = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %1
  %2 = load i32* %arrayidx, align 4
  %3 = load i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32* %i, align 4
  %sub = sub nsw i32 100, %4
  %sub1 = sub nsw i32 %sub, 1
  %arrayidx2 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %sub1
  store i32 %add, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32* %arrayidx3, align 4
  ret i32 %7
}

; int mayAlias03 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     *(PA+i) = *(PB+SIZE-i-1) + a;
;   return *(PA+a);
; }
; CHECK: define i32 @mayAlias03
; CHECK-NOT: add nsw <4 x i32>
; CHECK ret

define i32 @mayAlias03(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32** @PB, align 4
  %add.ptr = getelementptr inbounds i32* %1, i32 100
  %2 = load i32* %i, align 4
  %idx.neg = sub i32 0, %2
  %add.ptr1 = getelementptr inbounds i32* %add.ptr, i32 %idx.neg
  %add.ptr2 = getelementptr inbounds i32* %add.ptr1, i32 -1
  %3 = load i32* %add.ptr2, align 4
  %4 = load i32* %a.addr, align 4
  %add = add nsw i32 %3, %4
  %5 = load i32** @PA, align 4
  %6 = load i32* %i, align 4
  %add.ptr3 = getelementptr inbounds i32* %5, i32 %6
  store i32 %add, i32* %add.ptr3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i32** @PA, align 4
  %9 = load i32* %a.addr, align 4
  %add.ptr4 = getelementptr inbounds i32* %8, i32 %9
  %10 = load i32* %add.ptr4, align 4
  ret i32 %10
}

; int mustAlias01 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i+10] = Foo.B[SIZE-i-1] + a;
;   return Foo.A[a];
; }
; CHECK: define i32 @mustAlias01
; CHECK-NOT: add nsw <4 x i32>
; CHECK ret

define i32 @mustAlias01(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 1
  %arrayidx = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32* %arrayidx, align 4
  %3 = load i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32* %i, align 4
  %add2 = add nsw i32 %4, 10
  %arrayidx3 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %add2
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32* %a.addr, align 4
  %arrayidx4 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32* %arrayidx4, align 4
  ret i32 %7
}

; int mustAlias02 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i] = Foo.B[SIZE-i-10] + a;
;   return Foo.A[a];
; }
; CHECK: define i32 @mustAlias02
; CHECK-NOT: add nsw <4 x i32>
; CHECK ret

define i32 @mustAlias02(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 10
  %arrayidx = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32* %arrayidx, align 4
  %3 = load i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32* %i, align 4
  %arrayidx2 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %4
  store i32 %add, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32* %arrayidx3, align 4
  ret i32 %7
}

; int mustAlias03 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i+10] = Foo.B[SIZE-i-10] + a;
;   return Foo.A[a];
; }
; CHECK: define i32 @mustAlias03
; CHECK-NOT: add nsw <4 x i32>
; CHECK ret

define i32 @mustAlias03(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 10
  %arrayidx = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32* %arrayidx, align 4
  %3 = load i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32* %i, align 4
  %add2 = add nsw i32 %4, 10
  %arrayidx3 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %add2
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32* %a.addr, align 4
  %arrayidx4 = getelementptr inbounds [100 x i32]* getelementptr inbounds (%struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32* %arrayidx4, align 4
  ret i32 %7
}
