; RUN: llvm-extract -func foo -S < %s | FileCheck %s
; RUN: llvm-extract -delete -func foo -S < %s | FileCheck --check-prefix=DELETE %s
; RUN: llvm-as < %s > %t
; RUN: llvm-extract -func foo -S %t | FileCheck %s
; RUN: llvm-extract -delete -func foo -S %t | FileCheck --check-prefix=DELETE %s

; llvm-extract uses lazy bitcode loading, so make sure it correctly reads
; from bitcode files in addition to assembly files.

; CHECK: define void @foo() {
; CHECK:   ret void
; CHECK: }
; DELETE: define void @bar() {
; DELETE:   ret void
; DELETE: }

define void @foo() {
  ret void
}
define void @bar() {
  ret void
}
