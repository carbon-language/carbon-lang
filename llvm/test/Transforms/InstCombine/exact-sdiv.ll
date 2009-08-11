; RUN: llvm-as < %s | opt -instcombine | llvm-dis | FileCheck %s

; CHECK: define i32 @foo
; CHECK: sdiv i32 %x, 8
define i32 @foo(i32 %x) {
  %y = sdiv i32 %x, 8
  ret i32 %y
}

; CHECK: define i32 @bar
; CHECK: ashr i32 %x, 3
define i32 @bar(i32 %x) {
  %y = sdiv exact i32 %x, 8
  ret i32 %y
}
