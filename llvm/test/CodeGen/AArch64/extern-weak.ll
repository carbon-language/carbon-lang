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


@arr_var = extern_weak global [10 x i32]

define i32* @bar() {
  %addr = getelementptr [10 x i32]* @arr_var, i32 0, i32 5
; CHECK: .LCPI1_0:
; CHECK-NEXT: .xword arr_var

; CHECK: ldr [[BASE:x[0-9]+]], [{{x[0-9]+}}, #:lo12:.LCPI1_0]
; CHECK: add x0, [[BASE]], #20
  ret i32* %addr
}

@defined_weak_var = internal unnamed_addr global i32 0

define i32* @wibble() {
  ret i32* @defined_weak_var
; CHECK: adrp [[BASE:x[0-9]+]], defined_weak_var
; CHECK: add x0, [[BASE]], #:lo12:defined_weak_var
}