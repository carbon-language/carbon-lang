; RUN: opt %s -tailcallelim -S | FileCheck %s

declare void @use(i8* nocapture, i8* nocapture)
declare void @boring()

define i8* @test1(i8* nocapture %A, i1 %cond) {
; CHECK: tailrecurse:
; CHECK: %A.tr = phi i8* [ %A, %0 ], [ %B, %cond_true ]
; CHECK: %cond.tr = phi i1 [ %cond, %0 ], [ false, %cond_true ]
  %B = alloca i8
; CHECK: %B = alloca i8
  br i1 %cond, label %cond_true, label %cond_false
; CHECK: br i1 %cond.tr, label %cond_true, label %cond_false
cond_true:
; CHECK: cond_true:
; CHECK: br label %tailrecurse
  call i8* @test1(i8* %B, i1 false)
  ret i8* null
cond_false:
; CHECK: cond_false
  call void @use(i8* %A, i8* %B)
; CHECK: call void @use(i8* %A.tr, i8* %B)
  call void @boring()
; CHECK: tail call void @boring()
  ret i8* null
; CHECK: ret i8* null
}

; PR14143
define void @test2(i8* %a, i8* %b) {
; CHECK: @test2
; CHECK-NOT: tail call
; CHECK: ret void
  %c = alloca [100 x i8], align 16
  %tmp = bitcast [100 x i8]* %c to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %b, i8* %tmp, i64 100, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)
