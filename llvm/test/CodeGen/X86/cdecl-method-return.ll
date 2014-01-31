; RUN: llc < %s -mtriple=i686-pc-win32 -mcpu=core2 | FileCheck %s

; The sret flag causes the first two parameters to be reordered on the stack.

define x86_cdeclmethodcc void @foo(i32* sret %dst, i32* %src) {
  %v = load i32* %src
  store i32 %v, i32* %dst
  ret void
}

; CHECK-LABEL: _foo:
; CHECK:  movl    8(%esp), %[[dst:[^ ]*]]
; CHECK:  movl    4(%esp), %[[src:[^ ]*]]
; CHECK:  movl    (%[[src]]), %[[v:[^ ]*]]
; CHECK:  movl    %[[v]], (%[[dst]])
; CHECK:  retl

define i32 @bar() {
  %src = alloca i32
  %dst = alloca i32
  store i32 42, i32* %src
  call x86_cdeclmethodcc void @foo(i32* sret %dst, i32* %src)
  %v = load i32* %dst
  ret i32 %v
}

; CHECK-LABEL: _bar:
; CHECK:  movl    $42, [[src:[^,]*]]
; CHECK:  leal    [[src]], %[[reg:[^ ]*]]
; CHECK:  movl    %[[reg]], (%esp)
; CHECK:  leal    [[dst:[^,]*]], %[[reg:[^ ]*]]
; CHECK:  movl    %[[reg]], 4(%esp)
; CHECK:  calll   _foo
; CHECK:  movl    [[dst]], %eax
; CHECK:  retl

; If we don't have the sret flag, parameters are not reordered.

define x86_cdeclmethodcc void @baz(i32* %dst, i32* %src) {
  %v = load i32* %src
  store i32 %v, i32* %dst
  ret void
}

; CHECK-LABEL: _baz:
; CHECK:  movl    4(%esp), %[[dst:[^ ]*]]
; CHECK:  movl    8(%esp), %[[src:[^ ]*]]
; CHECK:  movl    (%[[src]]), %[[v:[^ ]*]]
; CHECK:  movl    %[[v]], (%[[dst]])
; CHECK:  retl

define i32 @qux() {
  %src = alloca i32
  %dst = alloca i32
  store i32 42, i32* %src
  call x86_cdeclmethodcc void @baz(i32* %dst, i32* %src)
  %v = load i32* %dst
  ret i32 %v
}

; CHECK-LABEL: _qux:
; CHECK:  movl    $42, [[src:[^,]*]]
; CHECK:  leal    [[src]], %[[reg:[^ ]*]]
; CHECK:  movl    %[[reg]], 4(%esp)
; CHECK:  leal    [[dst:[^,]*]], %[[reg:[^ ]*]]
; CHECK:  movl    %[[reg]], (%esp)
; CHECK:  calll   _baz
; CHECK:  movl    [[dst]], %eax
; CHECK:  retl
