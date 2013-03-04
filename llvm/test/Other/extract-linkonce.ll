; RUN: llvm-extract -func foo -S < %s | FileCheck %s
; RUN: llvm-extract -delete -func foo -S < %s | FileCheck --check-prefix=DELETE %s

; Test that we don't convert weak_odr to external definitions.

; CHECK:      @bar = external hidden global i32
; CHECK:      define hidden i32* @foo() {
; CHECK-NEXT:  ret i32* @bar
; CHECK-NEXT: }

; DELETE: @bar = hidden global i32 42
; DELETE: declare hidden i32* @foo()

@bar = linkonce global i32 42

define linkonce i32* @foo() {
  ret i32* @bar
}

define void @g() {
  call i32* @foo()
  ret void
}
