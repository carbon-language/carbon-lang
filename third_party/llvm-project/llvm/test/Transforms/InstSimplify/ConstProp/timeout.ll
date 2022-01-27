; NOTE: This is a timeout test for some O(something silly) constant folding behaviour. It may not be the best test. Providing it finishes, it passes.
; RUN: opt < %s -O3 -S | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8-none-eabi"

%struct.ST = type { %struct.ST* }

@global = internal global [121 x i8] zeroinitializer, align 1

define void @func() #0 {
;CHECK-LABEL: func
entry:
  %s = alloca %struct.ST*, align 4
  %j = alloca i32, align 4
  store %struct.ST* bitcast ([121 x i8]* @global to %struct.ST*), %struct.ST** %s, align 4
  store i32 0, i32* %j, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %j, align 4
  %cmp = icmp slt i32 %0, 30
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load %struct.ST*, %struct.ST** %s, align 4
  %2 = bitcast %struct.ST* %1 to i8*
  %add.ptr = getelementptr inbounds i8, i8* %2, i32 4
  %3 = ptrtoint i8* %add.ptr to i32
  %4 = load %struct.ST*, %struct.ST** %s, align 4
  %5 = bitcast %struct.ST* %4 to i8*
  %add.ptr1 = getelementptr inbounds i8, i8* %5, i32 4
  %6 = ptrtoint i8* %add.ptr1 to i32
  %rem = urem i32 %6, 2
  %cmp2 = icmp eq i32 %rem, 0
  br i1 %cmp2, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body
  br label %cond.end

cond.false:                                       ; preds = %for.body
  %7 = load %struct.ST*, %struct.ST** %s, align 4
  %8 = bitcast %struct.ST* %7 to i8*
  %add.ptr3 = getelementptr inbounds i8, i8* %8, i32 4
  %9 = ptrtoint i8* %add.ptr3 to i32
  %rem4 = urem i32 %9, 2
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 0, %cond.true ], [ %rem4, %cond.false ]
  %add = add i32 %3, %cond
  %10 = inttoptr i32 %add to %struct.ST*
  %11 = load %struct.ST*, %struct.ST** %s, align 4
  %next = getelementptr inbounds %struct.ST, %struct.ST* %11, i32 0, i32 0
  store %struct.ST* %10, %struct.ST** %next, align 4
  %12 = load %struct.ST*, %struct.ST** %s, align 4
  %next5 = getelementptr inbounds %struct.ST, %struct.ST* %12, i32 0, i32 0
  %13 = load %struct.ST*, %struct.ST** %next5, align 4
  store %struct.ST* %13, %struct.ST** %s, align 4
  br label %for.inc

for.inc:                                          ; preds = %cond.end
  %14 = load i32, i32* %j, align 4
  %inc = add nsw i32 %14, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %15 = load %struct.ST*, %struct.ST** %s, align 4
  %next6 = getelementptr inbounds %struct.ST, %struct.ST* %15, i32 0, i32 0
  store %struct.ST* null, %struct.ST** %next6, align 4
  ret void
}

