; RUN: opt < %s -loweratomic -S | FileCheck %s

declare i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* %ptr, i8 %cmp, i8 %val)
declare i8 @llvm.atomic.swap.i8.p0i8(i8* %ptr, i8 %val)

define i8 @cmpswap() {
; CHECK: @cmpswap
  %i = alloca i8
  %j = call i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* %i, i8 0, i8 42)
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}

define i8 @swap() {
; CHECK: @swap
  %i = alloca i8
  %j = call i8 @llvm.atomic.swap.i8.p0i8(i8* %i, i8 42)
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}
