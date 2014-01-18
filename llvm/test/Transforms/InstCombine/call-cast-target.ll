; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"

define i32 @main() {
; CHECK-LABEL: @main
; CHECK: call i32 bitcast
entry:
  %tmp = call i32 bitcast (i8* (i32*)* @ctime to i32 (i32*)*)( i32* null )          ; <i32> [#uses=1]
  ret i32 %tmp
}

declare i8* @ctime(i32*)

define internal { i8 } @foo(i32*) {
entry:
  ret { i8 } { i8 0 }
}

define void @test_struct_ret() {
; CHECK-LABEL: @test_struct_ret
; CHECK-NOT: bitcast
entry:
  %0 = call { i8 } bitcast ({ i8 } (i32*)* @foo to { i8 } (i16*)*)(i16* null)
  ret void
}
