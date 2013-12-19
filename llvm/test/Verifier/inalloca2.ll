; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @doit(i64* inalloca %a)

define void @a() {
entry:
  %a = alloca [2 x i32]
  %b = bitcast [2 x i32]* %a to i64*
  call void @doit(i64* inalloca %b)
; CHECK: Inalloca argument is not an alloca!
  ret void
}

define void @b() {
entry:
  %a = alloca i64
  call void @doit(i64* inalloca %a)
  call void @doit(i64* inalloca %a)
; CHECK: Allocas can be used at most once with inalloca!
  ret void
}
