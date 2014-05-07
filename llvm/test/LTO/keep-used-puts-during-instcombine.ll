; RUN: opt -S -instcombine <%s | FileCheck %s
; rdar://problem/16165191
; llvm.compiler.used functions should not be renamed

target triple = "x86_64-apple-darwin11"

@llvm.compiler.used = appending global [1 x i8*] [
  i8* bitcast (i32(i8*)* @puts to i8*)
  ], section "llvm.metadata"
@llvm.used = appending global [1 x i8*] [
  i8* bitcast (i32(i32)* @uses_printf to i8*)
  ], section "llvm.metadata"

@str = private unnamed_addr constant [13 x i8] c"hello world\0A\00"

define i32 @uses_printf(i32 %i) {
entry:
  %s = getelementptr [13 x i8]* @str, i64 0, i64 0
  call i32 (i8*, ...)* @printf(i8* %s)
  ret i32 0
}

define internal i32 @printf(i8* readonly nocapture %fmt, ...) {
entry:
  %ret = call i32 @bar(i8* %fmt)
  ret i32 %ret
}

; CHECK: define {{.*}} @puts(
define internal i32 @puts(i8* %s) {
entry:
  %ret = call i32 @bar(i8* %s)
  ret i32 %ret
}

declare i32 @bar(i8*)
