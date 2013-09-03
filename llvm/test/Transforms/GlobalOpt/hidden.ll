; RUN: opt %s -globalopt -S | FileCheck %s

@foo = linkonce_odr unnamed_addr constant i32 42
; CHECK: @foo = linkonce_odr hidden unnamed_addr constant i32 42

define linkonce_odr void @bar() unnamed_addr {
; CHECK: define linkonce_odr hidden void @bar() unnamed_addr {
  ret void
}

define i32* @zed() {
  call void @bar()
  ret i32* @foo
}
