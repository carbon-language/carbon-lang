; Test that llvm-reduce can remove uninteresting Global Variables as well as
; their direct uses.
;
; RUN: llvm-reduce --test %p/Inputs/remove-global-vars.sh %s
; RUN: cat reduced.ll | FileCheck %s
; REQUIRES: plugins, shell

@uninteresting1 = global i32 0, align 4
; CHECK: @interesting = global
@interesting = global i32 5, align 4
; CHECK-NOT: global
@uninteresting2 = global i32 25, align 4
@uninteresting3 = global i32 50, align 4

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ; CHECK-NOT: load i32, i32* @uninteresting2, align 4
  %0 = load i32, i32* @uninteresting2, align 4
  store i32 %0, i32* @interesting, align 4
  ; CHECK-NOT: load i32, i32* @uninteresting3, align 4
  %1 = load i32, i32* @uninteresting3, align 4
  %dec = add nsw i32 %1, -1
  ; CHECK-NOT: store i32 %dec, i32* @uninteresting3, align 4
  store i32 %dec, i32* @uninteresting3, align 4
  ; CHECK: load i32, i32* @interesting, align 4
  %2 = load i32, i32* @interesting, align 4
  ; CHECK-NOT: load i32, i32* @uninteresting2, align 4
  %3 = load i32, i32* @uninteresting2, align 4
  %add = add nsw i32 %2, %3
  ; CHECK-NOT: store i32 %add, i32* @uninteresting1, align 4
  store i32 %add, i32* @uninteresting1, align 4
  store i32 10, i32* @interesting, align 4
  ; CHECK: load i32, i32* @interesting, align 4
  %4 = load i32, i32* @interesting, align 4
  ret i32 0
}
