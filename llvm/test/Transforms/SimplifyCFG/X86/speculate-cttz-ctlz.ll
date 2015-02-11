; RUN: opt -S -simplifycfg -mtriple=x86_64-unknown-unknown -mattr=+bmi < %s | FileCheck %s --check-prefix=ALL --check-prefix=BMI
; RUN: opt -S -simplifycfg -mtriple=x86_64-unknown-unknown -mattr=+lzcnt < %s | FileCheck %s --check-prefix=ALL --check-prefix=LZCNT
; RUN: opt -S -simplifycfg -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefix=ALL --check-prefix=GENERIC


define i64 @test1(i64 %A) {
; ALL-LABEL: @test1(
; ALL: [[COND:%[A-Za-z0-9]+]] = icmp eq i64 %A, 0
; ALL: [[CTLZ:%[A-Za-z0-9]+]] = tail call i64 @llvm.ctlz.i64(i64 %A, i1 true)
; LZCNT-NEXT: select i1 [[COND]], i64 64, i64 [[CTLZ]]
; BMI-NOT: select
; GENERIC-NOT: select
; ALL: ret
entry:
  %tobool = icmp eq i64 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.ctlz.i64(i64 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i64 [ %0, %cond.true ], [ 64, %entry ]
  ret i64 %cond
}

define i32 @test2(i32 %A) {
; ALL-LABEL: @test2(
; ALL: [[COND:%[A-Za-z0-9]+]] = icmp eq i32 %A, 0
; ALL: [[CTLZ:%[A-Za-z0-9]+]] = tail call i32 @llvm.ctlz.i32(i32 %A, i1 true)
; LZCNT-NEXT: select i1 [[COND]], i32 32, i32 [[CTLZ]]
; BMI-NOT: select
; GENERIC-NOT: select
; ALL: ret
entry:
  %tobool = icmp eq i32 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.ctlz.i32(i32 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %0, %cond.true ], [ 32, %entry ]
  ret i32 %cond
}


define signext i16 @test3(i16 signext %A) {
; ALL-LABEL: @test3(
; ALL: [[COND:%[A-Za-z0-9]+]] = icmp eq i16 %A, 0
; ALL: [[CTLZ:%[A-Za-z0-9]+]] = tail call i16 @llvm.ctlz.i16(i16 %A, i1 true)
; LZCNT-NEXT: select i1 [[COND]], i16 16, i16 [[CTLZ]]
; BMI-NOT: select
; GENERIC-NOT: select
; ALL: ret
entry:
  %tobool = icmp eq i16 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i16 @llvm.ctlz.i16(i16 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %0, %cond.true ], [ 16, %entry ]
  ret i16 %cond
}


define i64 @test1b(i64 %A) {
; ALL-LABEL: @test1b(
; ALL: [[COND:%[A-Za-z0-9]+]] = icmp eq i64 %A, 0
; ALL: [[CTTZ:%[A-Za-z0-9]+]] = tail call i64 @llvm.cttz.i64(i64 %A, i1 true)
; BMI-NEXT: select i1 [[COND]], i64 64, i64 [[CTTZ]]
; LZCNT-NOT: select
; GENERIC-NOT: select
; ALL: ret
entry:
  %tobool = icmp eq i64 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.cttz.i64(i64 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i64 [ %0, %cond.true ], [ 64, %entry ]
  ret i64 %cond
}


define i32 @test2b(i32 %A) {
; ALL-LABEL: @test2b(
; ALL: [[COND:%[A-Za-z0-9]+]] = icmp eq i32 %A, 0
; ALL: [[CTTZ:%[A-Za-z0-9]+]] = tail call i32 @llvm.cttz.i32(i32 %A, i1 true)
; BMI-NEXT: select i1 [[COND]], i32 32, i32 [[CTTZ]]
; LZCNT-NOT: select
; GENERIC-NOT: select
; ALL: ret
entry:
  %tobool = icmp eq i32 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.cttz.i32(i32 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %0, %cond.true ], [ 32, %entry ]
  ret i32 %cond
}


define signext i16 @test3b(i16 signext %A) {
; ALL-LABEL: @test3b(
; ALL: [[COND:%[A-Za-z0-9]+]] = icmp eq i16 %A, 0
; ALL: [[CTTZ:%[A-Za-z0-9]+]] = tail call i16 @llvm.cttz.i16(i16 %A, i1 true)
; BMI-NEXT: select i1 [[COND]], i16 16, i16 [[CTTZ]]
; LZCNT-NOT: select
; GENERIC-NOT: select
; ALL: ret
entry:
  %tobool = icmp eq i16 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i16 @llvm.cttz.i16(i16 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %0, %cond.true ], [ 16, %entry ]
  ret i16 %cond
}

declare i64 @llvm.ctlz.i64(i64, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i16 @llvm.ctlz.i16(i16, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i16 @llvm.cttz.i16(i16, i1)
