; RUN: opt -licm -basic-aa < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s

define void @test1(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test1(
; CHECK-LABEL: entry:
; CHECK: call {}* @llvm.invariant.start.p0i32(i64 4, i32* %ptr)
; CHECK: %val = load i32, i32* %ptr
; CHECK-LABEL: loop:

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %ptr)
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}

;; despite the loop varying invariant.start, we should be
;; able to hoist the load
define void @test2(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test2(
; CHECK-LABEL: entry:
; CHECK: %val = load i32, i32* %ptr
; CHECK-LABEL: loop:
; CHECK: call {}* @llvm.invariant.start.p0i32(i64 4, i32* %piv)
entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  %piv = getelementptr i32, i32* %ptr, i32 %x
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %piv)
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}

define void @test3(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test3(
; CHECK-LABEL: entry:
; CHECK: call {}* @llvm.invariant.start.p0i32(i64 4, i32* %ptr)
; CHECK: %val = load i32, i32* %ptr
; CHECK-LABEL: loop:
entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %ptr)
  %val = load i32, i32* %ptr
  %p2 = getelementptr i32, i32* %ptr, i32 1
  store volatile i32 0, i32* %p2
  %x.inc = add i32 %x, %val
  br label %loop
}

; can't hoist due to init in loop, only well defined if loop exits
; on first iteration, but we don't bother checking for that currently
define void @test4(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test4(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK:   store i32 0, i32* %ptr
; CHECK: call {}* @llvm.invariant.start.p0i32(i64 4, i32* %ptr)
; CHECK: %val = load i32, i32* %ptr
entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  store i32 0, i32* %ptr
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %ptr)
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}

; don't try to reason about scopes
define void @test5(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test5(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK:   store i32 0, i32* %ptr
; CHECK: call {}* @llvm.invariant.start.p0i32(i64 4, i32* %ptr)
; CHECK: %val = load i32, i32* %ptr
entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  store i32 0, i32* %ptr
  %scope = call {}* @llvm.invariant.start.p0i32(i64 4, i32* %ptr)
  %val = load i32, i32* %ptr
  call void @llvm.invariant.end.p0i32({}* %scope, i64 4, i32* %ptr)
  %x.inc = add i32 %x, %val
  br label %loop
}

declare {}* @llvm.invariant.start.p0i32(i64, i32*)
declare void @llvm.invariant.end.p0i32({}*, i64, i32*)
