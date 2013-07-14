; RUN: opt < %s -instsimplify -S | FileCheck %s

define i1 @add(i1 %x) {
; CHECK-LABEL: @add(
  %z = add i1 %x, %x
  ret i1 %z
; CHECK: ret i1 false
}

define i1 @sub(i1 %x) {
; CHECK-LABEL: @sub(
  %z = sub i1 false, %x
  ret i1 %z
; CHECK: ret i1 %x
}

define i1 @mul(i1 %x) {
; CHECK-LABEL: @mul(
  %z = mul i1 %x, %x
  ret i1 %z
; CHECK: ret i1 %x
}

define i1 @ne(i1 %x) {
; CHECK-LABEL: @ne(
  %z = icmp ne i1 %x, 0
  ret i1 %z
; CHECK: ret i1 %x
}
