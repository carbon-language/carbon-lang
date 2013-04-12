; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs -filetype=obj < %s | llvm-readobj -s -r | FileCheck %s

define i64 @testfn() nounwind {
entry:
  ret i64 0
}

define i64 @foo() nounwind {
entry:
  %bar = alloca i64 ()*, align 8
  store i64 ()* @testfn, i64 ()** %bar, align 8
  %call = call i64 @testfn()
  ret i64 %call
}

; The above should produce an ADRP/ADD pair to calculate the address of
; testfn. The important point is that LLVM shouldn't think it can deal with the
; relocation on the ADRP itself (even though it knows everything about the
; relative offsets of testfn and foo) because its value depends on where this
; object file's .text section gets relocated in memory.

; CHECK:      Relocations [
; CHECK-NEXT:   Section (1) .text {
; CHECK-NEXT:     0x10 R_AARCH64_ADR_PREL_PG_HI21 testfn 0x0
; CHECK-NEXT:     0x14 R_AARCH64_ADD_ABS_LO12_NC testfn 0x0
; CHECK-NEXT:   }
; CHECK-NEXT: ]
