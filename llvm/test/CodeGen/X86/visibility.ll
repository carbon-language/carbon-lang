; RUN: llc  %s -o - | FileCheck %s

define hidden void @foo() nounwind {
entry:
  call void @bar()
  ret void
}

declare hidden void @bar()

;CHECK: .hidden	bar
