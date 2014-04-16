; RUN: llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=arm64-apple-ios7.0 -o - %s | FileCheck %s
@var = global void()* zeroinitializer

declare void @bar()

define void @foo() {
; CHECK-LABEL: foo:
       %func = load void()** @var

       ; Calling a function encourages @foo to use a callee-saved register,
       ; which makes it a natural choice for the tail call itself. But we don't
       ; want that: the final "br xN" has to use a temporary or argument
       ; register.
       call void @bar()

       tail call void %func()
; CHECK: br {{x([0-79]|1[0-8])}}
       ret void
}
