; RUN: llc -mtriple=aarch64-none-linux-gnu -o - < %s | FileCheck %s

declare extern_weak i32 @var()

define i32()* @foo() {
; The usual ADRP/ADD pair can't be used for a weak reference because it must
; evaluate to 0 if the symbol is undefined. We use a litpool entry.
  ret i32()* @var
; CHECK: .LCPI0_0:
; CHECK-NEXT: .xword var

; CHECK: ldr x0, [{{x[0-9]+}}, #:lo12:.LCPI0_0]

}
