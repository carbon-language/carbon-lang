; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; CHECK: // .weak foo
; CHECK: .weak .func foo
define weak void @foo() {
  ret void
}

; CHECK: // .weak baz
; CHECK: .weak .func baz
define weak_odr void @baz() {
  ret void
}

; CHECK: .visible .func bar
define void @bar() {
  ret void
}
