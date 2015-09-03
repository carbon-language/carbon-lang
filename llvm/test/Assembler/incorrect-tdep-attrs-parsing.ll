; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: define i32 @f(i64 "foo bar", i64, i64, i64 "xyz") {
define i32 @f(i64 "foo bar", i64, i64, i64 "xyz") {
  ret i32 41
}
