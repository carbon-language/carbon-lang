; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: echo "VERSION_1.0{ global: foo; local: *; }; VERSION_2.0{ global: bar; local: *; };" > %t.script
; RUN: ld.lld %t.o -o %t2 -shared --version-script %t.script -save-temps
; RUN: llvm-dis < %t2.0.0.preopt.bc | FileCheck %s
; RUN: llvm-readobj --dyn-syms %t2 | FileCheck --check-prefix=DSO %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

; CHECK: define void @foo()
; CHECK: define void @bar()

; DSO: DynamicSymbols [
; DSO:   Symbol {
; DSO:     Name:
; DSO:     Value: 0x0
; DSO:     Size: 0
; DSO:     Binding: Local
; DSO:     Type: None
; DSO:     Other: 0
; DSO:     Section: Undefined
; DSO:   }
; DSO:   Symbol {
; DSO:     Name: foo@@VERSION_1.0
; DSO:     Value:
; DSO:     Size: 1
; DSO:     Binding: Global
; DSO:     Type: Function
; DSO:     Other: 0
; DSO:     Section: .text
; DSO:   }
; DSO:   Symbol {
; DSO:     Name: bar@@VERSION_2.0
; DSO:     Value:
; DSO:     Size: 1
; DSO:     Binding: Global
; DSO:     Type: Function
; DSO:     Other: 0
; DSO:     Section: .text
; DSO:   }
; DSO: ]
