; REQUIRES: asserts
; RUN: llc -mtriple=thumbv7-none-linux-gnueabi -debug -o /dev/null < %s 2>&1 | FileCheck %s

; This test makes sure spills of 64-bit pairs in Thumb mode actually
; generate thumb instructions. Previously we were inserting an ARM
; STMIA which happened to have the same encoding.

define void @foo(i64* %addr) {
  %val1 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(i64* %addr)
  %val2 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(i64* %addr)
  %val3 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(i64* %addr)
  %val4 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(i64* %addr)
  %val5 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(i64* %addr)
  %val6 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(i64* %addr)
  %val7 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(i64* %addr)

  ; Make sure we are actually creating the Thumb versions of the spill
  ; instructions.
; CHECK: t2STRDi8
; CHECK: t2LDRDi8

  store volatile i64 %val1, i64* %addr
  store volatile i64 %val2, i64* %addr
  store volatile i64 %val3, i64* %addr
  store volatile i64 %val4, i64* %addr
  store volatile i64 %val5, i64* %addr
  store volatile i64 %val6, i64* %addr
  store volatile i64 %val7, i64* %addr
  ret void
}
