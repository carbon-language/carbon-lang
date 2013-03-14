; RUN: llc < %s -fast-isel -verify-machineinstrs -mtriple=x86_64-apple-darwin10
; RUN: llc < %s -fast-isel -verify-machineinstrs -mtriple=x86_64-pc-win32 | FileCheck %s 
; Requires: Asserts

; Previously, this would cause an assert.
define i31 @t1(i31 %a, i31 %b, i31 %c) {
entry:
  %add = add nsw i31 %b, %a
  %add1 = add nsw i31 %add, %c
  ret i31 %add1
}

; We don't handle the Windows CC, yet.
define i32 @foo(i32* %p) {
entry:
; CHECK: foo
; CHECK: movl (%rcx), %eax
  %0 = load i32* %p, align 4
  ret i32 %0
}
