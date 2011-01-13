; RUN: llvm-link %s %p/unnamed-addr2-b.ll -S -o - | FileCheck %s

define i32 @bar() {
entry:
  %call = tail call i32 @foo()
  ret i32 %call
}

declare i32 @foo()

; CHECK: define unnamed_addr i32 @foo()
