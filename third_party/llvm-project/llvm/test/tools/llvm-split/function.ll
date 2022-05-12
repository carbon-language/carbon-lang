; RUN: llvm-split -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0: define void @foo()
; CHECK1: declare void @foo()
define void @foo() {
  call void @bar()
  ret void
}

; CHECK0: declare void @bar()
; CHECK1: define void @bar()
define void @bar() {
  call void @foo()
  ret void
}
