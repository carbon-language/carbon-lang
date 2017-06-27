; RUN: opt < %s -loop-extract -S | FileCheck %s

@choum.addr = internal unnamed_addr constant [3 x i8*] [i8* blockaddress(@choum, %12), i8* blockaddress(@choum, %16), i8* blockaddress(@choum, %20)]

; CHECK: define
; no outlined function
; CHECK-NOT: define

define void @choum(i32, i32* nocapture, i32) {
  %4 = icmp sgt i32 %0, 0
  br i1 %4, label %5, label %26

  %6 = sext i32 %2 to i64
  %7 = getelementptr inbounds [3 x i8*], [3 x i8*]* @choum.addr, i64 0, i64 %6
  %8 = load i8*, i8** %7
  %9 = zext i32 %0 to i64
  br label %10

  %11 = phi i64 [ 0, %5 ], [ %24, %20 ]
  indirectbr i8* %8, [label %12, label %16, label %20]

  %13 = getelementptr inbounds i32, i32* %1, i64 %11
  %14 = load i32, i32* %13
  %15 = add nsw i32 %14, 1
  store i32 %15, i32* %13
  br label %16

  %17 = getelementptr inbounds i32, i32* %1, i64 %11
  %18 = load i32, i32* %17
  %19 = shl nsw i32 %18, 1
  store i32 %19, i32* %17
  br label %20

  %21 = getelementptr inbounds i32, i32* %1, i64 %11
  %22 = load i32, i32* %21
  %23 = add nsw i32 %22, -3
  store i32 %23, i32* %21
  %24 = add nuw nsw i64 %11, 1
  %25 = icmp eq i64 %24, %9
  br i1 %25, label %26, label %10

  ret void
}
