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

; The linkonce_odr linkage for foo() should be changed to external linkage.
; DELETE: declare void @foo()
; DELETE: define void @bar() {
; DELETE:   call void @foo()
; DELETE:   ret void
; DELETE: }

define linkonce_odr void @foo() {
  ret void
}
define void @bar() {
  call void @foo()
  ret void
}
