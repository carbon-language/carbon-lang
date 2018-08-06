; RUN: opt -licm -basicaa < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' < %s -S | FileCheck %s

; TODO: should be able to hoist both load and invariant.start
define void @test1(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test1(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK: call {}* @llvm.invariant.start.p0i32(i64 4, i32* %ptr)
; CHECK: %val = load i32, i32* %ptr

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %ptr)
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}

declare {}* @llvm.invariant.start.p0i32(i64, i32*)
