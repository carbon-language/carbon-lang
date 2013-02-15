; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs -filetype=obj < %s | elf-dump | FileCheck %s

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

; CHECK: .rela.text

; CHECK: # Relocation 0
; CHECK-NEXT: (('r_offset', 0x0000000000000010)
; CHECK-NEXT:  ('r_sym', 0x00000007)
; CHECK-NEXT:  ('r_type', 0x00000113)
; CHECK-NEXT:  ('r_addend', 0x0000000000000000)
; CHECK-NEXT: ),
; CHECK-NEXT:  Relocation 1
; CHECK-NEXT: (('r_offset', 0x0000000000000014)
; CHECK-NEXT:  ('r_sym', 0x00000007)
; CHECK-NEXT:  ('r_type', 0x00000115)
; CHECK-NEXT:  ('r_addend', 0x0000000000000000)
; CHECK-NEXT: ),
