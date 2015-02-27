; RUN: llc < %s -mtriple=x86_64-linux-gnux32 -O0 | FileCheck %s
; CHECK: leal {{[-0-9]*}}(%r{{s|b}}p),
; CHECK-NOT: leal {{[-0-9]*}}(%e{{s|b}}p),

define void @foo(i32** %p) {
  %a = alloca i32, i32 10
  %addr = getelementptr i32, i32* %a, i32 4
  store i32* %addr, i32** %p
  ret void
}
