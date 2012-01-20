; RUN: opt -inline < %s -S -o - -inline-threshold=8 | FileCheck %s

declare void @llvm.lifetime.start(i64 %size, i8* nocapture %ptr)

@glbl = external global i32

define void @outer1() {
; CHECK: @outer1
; CHECK-NOT: call void @inner1
  %ptr = alloca i32
  call void @inner1(i32* %ptr)
  ret void
}

define void @inner1(i32 *%ptr) {
  %A = load i32* %ptr
  store i32 0, i32* %ptr
  %C = getelementptr i32* %ptr, i32 0
  %D = getelementptr i32* %ptr, i32 1
  %E = bitcast i32* %ptr to i8*
  %F = select i1 false, i32* %ptr, i32* @glbl
  call void @llvm.lifetime.start(i64 0, i8* %E)
  ret void
}

define void @outer2() {
; CHECK: @outer2
; CHECK: call void @inner2
  %ptr = alloca i32
  call void @inner2(i32* %ptr)
  ret void
}

; %D poisons this call, scalar-repl can't handle that instruction.
define void @inner2(i32 *%ptr) {
  %A = load i32* %ptr
  store i32 0, i32* %ptr
  %C = getelementptr i32* %ptr, i32 0
  %D = getelementptr i32* %ptr, i32 %A
  %E = bitcast i32* %ptr to i8*
  %F = select i1 false, i32* %ptr, i32* @glbl
  call void @llvm.lifetime.start(i64 0, i8* %E)
  ret void
}
