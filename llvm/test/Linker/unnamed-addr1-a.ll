; RUN: llvm-link %s %p/unnamed-addr1-b.ll -S -o - | FileCheck %s

@foo = external global i32

define i32 @bar() {
entry:
  %tmp = load i32* @foo, align 4
  ret i32 %tmp
}

; CHECK: @foo = common unnamed_addr global i32 0, align 4
