; RUN: opt < %s -functionattrs -S | FileCheck %s
; PR8279

@g = constant i32 1

define void @foo() {
; CHECK: void @foo() {
  %tmp = volatile load i32* @g
  ret void
}
