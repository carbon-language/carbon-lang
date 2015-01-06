; RUN: opt < %s -globalopt -S | FileCheck %s

$c = comdat any
; CHECK: $c = comdat any

define linkonce_odr void @foo() comdat($c) {
  ret void
}
; CHECK: define linkonce_odr void @foo() comdat($c)

define linkonce_odr void @bar() comdat($c) {
  ret void
}
; CHECK: define linkonce_odr void @bar() comdat($c)

define void @zed()  {
  call void @foo()
  ret void
}
