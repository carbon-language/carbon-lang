; RUN: llvm-link -S %s %p/Inputs/pr27044.ll -o - | FileCheck %s

; CHECK: define i32 @f1() {
; CHECK: define void @f2() comdat($foo) {

define i32 @f1() {
  ret i32 0
}
