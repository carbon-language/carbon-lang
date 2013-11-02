; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 -dso-symbol=foo1 -dso-symbol=foo2 -dso-symbol=foo3 \
; RUN:     -dso-symbol=foo4  %t1 -disable-opt
; RUN: llvm-nm %t2 | FileCheck %s

; FIXME: llvm-nm is printing 'd' instead of 't' for foo1.
; XFAIL: powerpc64

; CHECK: t foo1
define linkonce_odr void @"\01foo1"() noinline {
  ret void
}

; CHECK: {{W|T|S}} foo2
define linkonce_odr void @"\01foo2"() noinline {
  ret void
}

; CHECK: t foo3
define linkonce_odr void @"\01foo3"() noinline {
  ret void
}

; CHECK: {{W|T|S}} foo4
define linkonce_odr void @"\01foo4"() noinline {
  ret void
}

declare void @f(void()*)

declare void @p()

define void @bar() {
bb0:
  call void @"\01foo1"()
  call void @f(void()* @"\01foo2")
  invoke void @"\01foo3"() to label %bb1 unwind label %clean
bb1:
  invoke void @f(void()* @"\01foo4") to label %bb2 unwind label %clean
bb2:
  ret void
clean:
  landingpad {i32, i32} personality void()* @p cleanup
  ret void
}
