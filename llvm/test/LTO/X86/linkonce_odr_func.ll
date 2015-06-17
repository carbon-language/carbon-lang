; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 -dso-symbol=foo1 -dso-symbol=foo2 -dso-symbol=foo3 \
; RUN:     -dso-symbol=foo4 -dso-symbol=v1 -dso-symbol=v2 %t1 -O0
; RUN: llvm-nm %t2 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: t foo1
define linkonce_odr void @foo1() noinline {
  ret void
}

; CHECK: W foo2
define linkonce_odr void @foo2() noinline {
  ret void
}

; CHECK: t foo3
define linkonce_odr void @foo3() noinline {
  ret void
}

; CHECK: W foo4
define linkonce_odr void @foo4() noinline {
  ret void
}

; CHECK: r v1
@v1 = linkonce_odr constant i32 32

define i32 @useV1() {
  %x = load i32, i32* @v1
  ret i32 %x
}

; CHECK: V v2
@v2 = linkonce_odr global i32 32

define i32 @useV2() {
  %x = load i32, i32* @v2
  ret i32 %x
}

declare void @f(void()*)

declare void @p()

define void @bar() personality void()* @p {
bb0:
  call void @foo1()
  call void @f(void()* @foo2)
  invoke void @foo3() to label %bb1 unwind label %clean
bb1:
  invoke void @f(void()* @foo4) to label %bb2 unwind label %clean
bb2:
  ret void
clean:
  landingpad {i32, i32} cleanup
  ret void
}
