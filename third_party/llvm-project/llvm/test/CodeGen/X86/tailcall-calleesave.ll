; RUN: llc -tailcallopt -mcpu=core < %s | FileCheck %s

target triple = "i686-apple-darwin"

declare fastcc void @foo(i32, i32, i32, i32, i32, i32)
declare i32* @bar(i32*)

define fastcc void @hoge(i32 %b) nounwind {
; Do not overwrite pushed callee-save registers
; CHECK: pushl
; CHECK: subl $[[SIZE:[0-9]+]], %esp
; CHECK-NOT: [[SIZE]](%esp)
  %a = alloca i32
  store i32 0, i32* %a
  %d = tail call i32* @bar(i32* %a) nounwind
  store i32 %b, i32* %d
  tail call fastcc void @foo(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6) nounwind
  ret void
}
