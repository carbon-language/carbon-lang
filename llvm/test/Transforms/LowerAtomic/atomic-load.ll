; RUN: opt < %s -loweratomic -S | FileCheck %s

declare i8 @llvm.atomic.load.add.i8.p0i8(i8* %ptr, i8 %delta)
declare i8 @llvm.atomic.load.nand.i8.p0i8(i8* %ptr, i8 %delta)
declare i8 @llvm.atomic.load.min.i8.p0i8(i8* %ptr, i8 %delta)

define i8 @add() {
; CHECK: @add
  %i = alloca i8
  %j = call i8 @llvm.atomic.load.add.i8.p0i8(i8* %i, i8 42)
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: add
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}

define i8 @nand() {
; CHECK: @nand
  %i = alloca i8
  %j = call i8 @llvm.atomic.load.nand.i8.p0i8(i8* %i, i8 42)
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: and
; CHECK-NEXT: xor
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}

define i8 @min() {
; CHECK: @min
  %i = alloca i8
  %j = call i8 @llvm.atomic.load.min.i8.p0i8(i8* %i, i8 42)
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}
