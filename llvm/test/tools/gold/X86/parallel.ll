; RUN: llvm-as -o %t.bc %s
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so -u foo -u bar -plugin-opt jobs=2 -plugin-opt save-temps -o %t %t.bc
; RUN: llvm-nm %t.o0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-nm %t.o1 | FileCheck --check-prefix=CHECK1 %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK0-NOT: bar
; CHECK0: T foo
; CHECK0-NOT: bar
define void @foo() {
  call void @bar()
  ret void
}

; CHECK1-NOT: foo
; CHECK1: T bar
; CHECK1-NOT: foo
define void @bar() {
  call void @foo()
  ret void
}
