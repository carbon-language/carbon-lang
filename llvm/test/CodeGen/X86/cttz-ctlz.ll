; RUN: opt -S -codegenprepare -mtriple=x86_64-unknown-unknown -mattr=+bmi < %s | FileCheck %s --check-prefix=ALL --check-prefix=BMI
; RUN: opt -S -codegenprepare -mtriple=x86_64-unknown-unknown -mattr=+lzcnt < %s | FileCheck %s --check-prefix=ALL --check-prefix=LZCNT
; RUN: opt -S -codegenprepare -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefix=ALL --check-prefix=GENERIC


define i64 @test1(i64 %A) {
; ALL-LABEL: @test1(
; LZCNT: [[CTLZ:%[A-Za-z0-9]+]] = call i64 @llvm.ctlz.i64(i64 %A, i1 false)
; LZCNT-NEXT: ret i64 [[CTLZ]]
; BMI: icmp eq i64 %A, 0
; BMI: call i64 @llvm.ctlz.i64(i64 %A, i1 true)
; GENERIC: icmp eq i64 %A, 0
; GENERIC: call i64 @llvm.ctlz.i64(i64 %A, i1 true)
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
; LZCNT: [[CTLZ:%[A-Za-z0-9]+]] = call i32 @llvm.ctlz.i32(i32 %A, i1 false)
; LZCNT-NEXT: ret i32 [[CTLZ]]
; BMI: icmp eq i32 %A, 0
; BMI: call i32 @llvm.ctlz.i32(i32 %A, i1 true)
; GENERIC: icmp eq i32 %A, 0
; GENERIC: call i32 @llvm.ctlz.i32(i32 %A, i1 true)
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
; LZCNT: [[CTLZ:%[A-Za-z0-9]+]] = call i16 @llvm.ctlz.i16(i16 %A, i1 false)
; LZCNT-NEXT: ret i16 [[CTLZ]]
; BMI: icmp eq i16 %A, 0
; BMI: call i16 @llvm.ctlz.i16(i16 %A, i1 true)
; GENERIC: icmp eq i16 %A, 0
; GENERIC: call i16 @llvm.ctlz.i16(i16 %A, i1 true)
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
; LZCNT: icmp eq i64 %A, 0
; LZCNT: call i64 @llvm.cttz.i64(i64 %A, i1 true)
; BMI: [[CTTZ:%[A-Za-z0-9]+]] = call i64 @llvm.cttz.i64(i64 %A, i1 false)
; BMI-NEXT: ret i64 [[CTTZ]]
; GENERIC: icmp eq i64 %A, 0
; GENERIC: call i64 @llvm.cttz.i64(i64 %A, i1 true)
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
; LZCNT: icmp eq i32 %A, 0
; LZCNT: call i32 @llvm.cttz.i32(i32 %A, i1 true)
; BMI: [[CTTZ:%[A-Za-z0-9]+]] = call i32 @llvm.cttz.i32(i32 %A, i1 false)
; BMI-NEXT: ret i32 [[CTTZ]]
; GENERIC: icmp eq i32 %A, 0
; GENERIC: call i32 @llvm.cttz.i32(i32 %A, i1 true)
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
; LZCNT: icmp eq i16 %A, 0
; LZCNT: call i16 @llvm.cttz.i16(i16 %A, i1 true)
; BMI: [[CTTZ:%[A-Za-z0-9]+]] = call i16 @llvm.cttz.i16(i16 %A, i1 false)
; BMI-NEXT: ret i16 [[CTTZ]]
; GENERIC: icmp eq i16 %A, 0
; GENERIC: call i16 @llvm.cttz.i16(i16 %A, i1 true)
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


define i64 @test1c(i64 %A) {
; ALL-LABEL: @test1c(
; ALL: icmp eq i64 %A, 0
; ALL: call i64 @llvm.ctlz.i64(i64 %A, i1 true)
entry:
  %tobool = icmp eq i64 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.ctlz.i64(i64 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i64 [ %0, %cond.true ], [ 63, %entry ]
  ret i64 %cond
}

define i32 @test2c(i32 %A) {
; ALL-LABEL: @test2c(
; ALL: icmp eq i32 %A, 0
; ALL: call i32 @llvm.ctlz.i32(i32 %A, i1 true)
entry:
  %tobool = icmp eq i32 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.ctlz.i32(i32 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %0, %cond.true ], [ 31, %entry ]
  ret i32 %cond
}


define signext i16 @test3c(i16 signext %A) {
; ALL-LABEL: @test3c(
; ALL: icmp eq i16 %A, 0
; ALL: call i16 @llvm.ctlz.i16(i16 %A, i1 true)
entry:
  %tobool = icmp eq i16 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i16 @llvm.ctlz.i16(i16 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %0, %cond.true ], [ 15, %entry ]
  ret i16 %cond
}


define i64 @test1d(i64 %A) {
; ALL-LABEL: @test1d(
; ALL: icmp eq i64 %A, 0
; ALL: call i64 @llvm.cttz.i64(i64 %A, i1 true)
entry:
  %tobool = icmp eq i64 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.cttz.i64(i64 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i64 [ %0, %cond.true ], [ 63, %entry ]
  ret i64 %cond
}


define i32 @test2d(i32 %A) {
; ALL-LABEL: @test2d(
; ALL: icmp eq i32 %A, 0
; ALL: call i32 @llvm.cttz.i32(i32 %A, i1 true)
entry:
  %tobool = icmp eq i32 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.cttz.i32(i32 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %0, %cond.true ], [ 31, %entry ]
  ret i32 %cond
}


define signext i16 @test3d(i16 signext %A) {
; ALL-LABEL: @test3d(
; ALL: icmp eq i16 %A, 0
; ALL: call i16 @llvm.cttz.i16(i16 %A, i1 true)
entry:
  %tobool = icmp eq i16 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i16 @llvm.cttz.i16(i16 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %0, %cond.true ], [ 15, %entry ]
  ret i16 %cond
}

; The following tests verify that calls to cttz/ctlz are speculated even if
; basic block %cond.true has an extra zero extend/truncate which is "free"
; for the target.

define i64 @test1e(i32 %x) {
; ALL-LABEL: @test1e(
; LZCNT: icmp eq i32 %x, 0
; LZCNT: call i32 @llvm.cttz.i32(i32 %x, i1 true)
; BMI: call i32 @llvm.cttz.i32(i32 %x, i1 false)
; GENERIC: icmp eq i32 %x, 0
; GENERIC: call i32 @llvm.cttz.i32(i32 %x, i1 true)
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 true)
  %phitmp2 = zext i32 %0 to i64
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i64 [ %phitmp2, %cond.true ], [ 32, %entry ]
  ret i64 %cond
}

define i32 @test2e(i64 %x) {
; ALL-LABEL: @test2e(
; LZCNT: icmp eq i64 %x, 0
; LZCNT: call i64 @llvm.cttz.i64(i64 %x, i1 true)
; BMI: call i64 @llvm.cttz.i64(i64 %x, i1 false)
; GENERIC: icmp eq i64 %x, 0
; GENERIC: call i64 @llvm.cttz.i64(i64 %x, i1 true)
entry:
  %tobool = icmp eq i64 %x, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.cttz.i64(i64 %x, i1 true)
  %cast = trunc i64 %0 to i32
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %cast, %cond.true ], [ 64, %entry ]
  ret i32 %cond
}

define i64 @test3e(i32 %x) {
; ALL-LABEL: @test3e(
; BMI: icmp eq i32 %x, 0
; BMI: call i32 @llvm.ctlz.i32(i32 %x, i1 true)
; LZCNT: call i32 @llvm.ctlz.i32(i32 %x, i1 false)
; GENERIC: icmp eq i32 %x, 0
; GENERIC: call i32 @llvm.ctlz.i32(i32 %x, i1 true)
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  %phitmp2 = zext i32 %0 to i64
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i64 [ %phitmp2, %cond.true ], [ 32, %entry ]
  ret i64 %cond
}

define i32 @test4e(i64 %x) {
; ALL-LABEL: @test4e(
; BMI: icmp eq i64 %x, 0
; BMI: call i64 @llvm.ctlz.i64(i64 %x, i1 true)
; LZCNT: call i64 @llvm.ctlz.i64(i64 %x, i1 false)
; GENERIC: icmp eq i64 %x, 0
; GENERIC: call i64 @llvm.ctlz.i64(i64 %x, i1 true)
entry:
  %tobool = icmp eq i64 %x, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  %cast = trunc i64 %0 to i32
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %cast, %cond.true ], [ 64, %entry ]
  ret i32 %cond
}

define i16 @test5e(i64 %x) {
; ALL-LABEL: @test5e(
; BMI: icmp eq i64 %x, 0
; BMI: call i64 @llvm.ctlz.i64(i64 %x, i1 true)
; LZCNT: call i64 @llvm.ctlz.i64(i64 %x, i1 false)
; GENERIC: icmp eq i64 %x, 0
; GENERIC: call i64 @llvm.ctlz.i64(i64 %x, i1 true)
entry:
  %tobool = icmp eq i64 %x, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  %cast = trunc i64 %0 to i16
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %cast, %cond.true ], [ 64, %entry ]
  ret i16 %cond
}

define i16 @test6e(i32 %x) {
; ALL-LABEL: @test6e(
; BMI: icmp eq i32 %x, 0
; BMI: call i32 @llvm.ctlz.i32(i32 %x, i1 true)
; LZCNT: call i32 @llvm.ctlz.i32(i32 %x, i1 false)
; GENERIC: icmp eq i32 %x, 0
; GENERIC: call i32 @llvm.ctlz.i32(i32 %x, i1 true)
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  %cast = trunc i32 %0 to i16
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %cast, %cond.true ], [ 32, %entry ]
  ret i16 %cond
}

define i16 @test7e(i64 %x) {
; ALL-LABEL: @test7e(
; LZCNT: icmp eq i64 %x, 0
; LZCNT: call i64 @llvm.cttz.i64(i64 %x, i1 true)
; BMI: call i64 @llvm.cttz.i64(i64 %x, i1 false)
; GENERIC: icmp eq i64 %x, 0
; GENERIC: call i64 @llvm.cttz.i64(i64 %x, i1 true)
entry:
  %tobool = icmp eq i64 %x, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.cttz.i64(i64 %x, i1 true)
  %cast = trunc i64 %0 to i16
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %cast, %cond.true ], [ 64, %entry ]
  ret i16 %cond
}

define i16 @test8e(i32 %x) {
; ALL-LABEL: @test8e(
; LZCNT: icmp eq i32 %x, 0
; LZCNT: call i32 @llvm.cttz.i32(i32 %x, i1 true)
; BMI: call i32 @llvm.cttz.i32(i32 %x, i1 false)
; GENERIC: icmp eq i32 %x, 0
; GENERIC: call i32 @llvm.cttz.i32(i32 %x, i1 true)
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 true)
  %cast = trunc i32 %0 to i16
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %cast, %cond.true ], [ 32, %entry ]
  ret i16 %cond
}


declare i64 @llvm.ctlz.i64(i64, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i16 @llvm.ctlz.i16(i16, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i16 @llvm.cttz.i16(i16, i1)
