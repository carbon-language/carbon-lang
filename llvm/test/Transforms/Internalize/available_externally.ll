; RUN: opt < %s -internalize -internalize-public-api-list foo -S | FileCheck %s

; CHECK-LABEL: define void @foo(
define void @foo() {
  ret void
}

; CHECK-LABEL: define internal void @zed(
define void @zed() {
  ret void
}

; CHECK-LABEL: define available_externally void @bar(
define available_externally void @bar() {
  ret void
}
