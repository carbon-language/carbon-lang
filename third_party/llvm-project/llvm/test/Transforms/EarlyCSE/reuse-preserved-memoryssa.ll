; RUN: opt -memoryssa -gvn -early-cse-memssa %s -S | FileCheck %s

; CHECK: define void @foo(

define void @foo() {
  ret void
}
