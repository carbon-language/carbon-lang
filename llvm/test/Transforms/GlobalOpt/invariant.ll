; RUN: opt -globalopt -S -o - < %s | FileCheck %s

declare {}* @llvm.invariant.start(i64 %size, i8* nocapture %ptr)

define void @test1(i8* %ptr) {
  call {}* @llvm.invariant.start(i64 -1, i8* %ptr)
  ret void
}

@object1 = global i32 0
; CHECK: @object1 = constant i32 -1
define void @ctor1() {
  store i32 -1, i32* @object1
  %A = bitcast i32* @object1 to i8*
  call void @test1(i8* %A)
  ret void
}


@object2 = global i32 0
; CHECK: @object2 = global i32 0
define void @ctor2() {
  store i32 -1, i32* @object2
  %A = bitcast i32* @object2 to i8*
  %B = call {}* @llvm.invariant.start(i64 -1, i8* %A)
  %C = bitcast {}* %B to i8*
  ret void
}

@llvm.global_ctors = appending constant
  [2 x { i32, void ()* }]
  [ { i32, void ()* } { i32 65535, void ()* @ctor1 },
    { i32, void ()* } { i32 65535, void ()* @ctor2 } ]
