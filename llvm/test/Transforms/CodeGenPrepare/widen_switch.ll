;; PowerPC is arbitralily chosen as a 32/64-bit RISC representative to show the transform in all tests.
;; x86 is chosen to show that the transform may differ when 8-bit and 16-bit registers are available.

; RUN: opt < %s -codegenprepare -S -mtriple=powerpc64-unknown-unknown | FileCheck %s --check-prefix=PPC --check-prefix=ALL
; RUN: opt < %s -codegenprepare -S -mtriple=x86_64-unknown-unknown    | FileCheck %s --check-prefix=X86 --check-prefix=ALL

; PPC widens to 32-bit; no change for x86 because 16-bit registers are part of the architecture.

define i32 @widen_switch_i16(i32 %a)  {
entry:
  %trunc = trunc i32 %a to i16
  switch i16 %trunc, label %sw.default [
    i16 1, label %sw.bb0
    i16 -1, label %sw.bb1
  ]

sw.bb0:
  br label %return

sw.bb1:
  br label %return

sw.default:
  br label %return

return:
  %retval = phi i32 [ -1, %sw.default ], [ 0, %sw.bb0 ], [ 1, %sw.bb1 ]
  ret i32 %retval

; ALL-LABEL: @widen_switch_i16(
; PPC:       %0 = zext i16 %trunc to i32
; PPC-NEXT:  switch i32 %0, label %sw.default [
; PPC-NEXT:    i32 1, label %return
; PPC-NEXT:    i32 65535, label %sw.bb1
; 
; X86:       %trunc = trunc i32 %a to i16
; X86-NEXT:  switch i16 %trunc, label %sw.default [
; X86-NEXT:    i16 1, label %return
; X86-NEXT:    i16 -1, label %sw.bb1
}

; Both architectures widen to 32-bit from a smaller, non-native type.

define i32 @widen_switch_i17(i32 %a)  {
entry:
  %trunc = trunc i32 %a to i17
  switch i17 %trunc, label %sw.default [
    i17 10, label %sw.bb0
    i17 -1, label %sw.bb1
  ]

sw.bb0:
  br label %return

sw.bb1:
  br label %return

sw.default:
  br label %return

return:
  %retval = phi i32 [ -1, %sw.default ], [ 0, %sw.bb0 ], [ 1, %sw.bb1 ]
  ret i32 %retval

; ALL-LABEL: @widen_switch_i17(
; ALL:       %0 = zext i17 %trunc to i32
; ALL-NEXT:  switch i32 %0, label %sw.default [
; ALL-NEXT:    i32 10, label %return
; ALL-NEXT:    i32 131071, label %sw.bb1
}

; If the switch condition is a sign-extended function argument, then the
; condition and cases should be sign-extended rather than zero-extended
; because the sign-extension can be optimized away.

define i32 @widen_switch_i16_sext(i2 signext %a)  {
entry:
  switch i2 %a, label %sw.default [
    i2 1, label %sw.bb0
    i2 -1, label %sw.bb1
  ]

sw.bb0:
  br label %return

sw.bb1:
  br label %return

sw.default:
  br label %return

return:
  %retval = phi i32 [ -1, %sw.default ], [ 0, %sw.bb0 ], [ 1, %sw.bb1 ]
  ret i32 %retval

; ALL-LABEL: @widen_switch_i16_sext(
; PPC:       %0 = sext i2 %a to i32
; PPC-NEXT:  switch i32 %0, label %sw.default [
; PPC-NEXT:    i32 1, label %return
; PPC-NEXT:    i32 -1, label %sw.bb1
; 
; X86:       %0 = sext i2 %a to i8
; X86-NEXT:  switch i8 %0, label %sw.default [
; X86-NEXT:    i8 1, label %return
; X86-NEXT:    i8 -1, label %sw.bb1
}

