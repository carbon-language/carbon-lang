; RUN: opt < %s -internalize -internalize-dso-list foo1,foo2,foo3,foo4 -S | FileCheck %s

; CHECK: define internal void @foo1(
define linkonce_odr void @foo1() noinline {
  ret void
}

; CHECK: define linkonce_odr void @foo2(
define linkonce_odr void @foo2() noinline {
  ret void
}

; CHECK: define internal void @foo3(
define linkonce_odr void @foo3() noinline {
  ret void
}

; CHECK: define linkonce_odr void @foo4(
define linkonce_odr void @foo4() noinline {
  ret void
}

declare void @f(void()*)

define void @bar() {
bb0:
  call void @foo1()
  call void @f(void()* @foo2)
  invoke void @foo3() to label %bb1 unwind label %clean
bb1:
  invoke void @f(void()* @foo4) to label %bb2 unwind label %clean
bb2:
  ret void
clean:
  landingpad i32 personality i8* null cleanup
  ret void
}
