; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs -filetype=obj < %s | elf-dump | FileCheck %s

define fp128 @testfn() nounwind {
entry:
  ret fp128 0xL00000000000000004004500000000000
}

define fp128 @foo() nounwind {
entry:
  %bar = alloca fp128 ()*, align 8
  store fp128 ()* @testfn, fp128 ()** %bar, align 8
  %call = call fp128 @testfn()
  ret fp128 %call
}

; The above should produce an ADRP/ADD pair to calculate the address of
; testfn. The important point is that LLVM shouldn't think it can deal with the
; relocation on the ADRP itself (even though it knows everything about the
; relative offsets of testfn and foo) because its value depends on where this
; object file's .text section gets relocated in memory.

; CHECK: .rela.text

; CHECK: # Relocation 0
; CHECK-NEXT: (('r_offset', 0x0000000000000028)
; CHECK-NEXT:  ('r_sym', 0x00000009)
; CHECK-NEXT:  ('r_type', 0x00000113)
; CHECK-NEXT:  ('r_addend', 0x0000000000000000)
; CHECK-NEXT: ),
; CHECK-NEXT:  Relocation 1
; CHECK-NEXT: (('r_offset', 0x000000000000002c)
; CHECK-NEXT:  ('r_sym', 0x00000009)
; CHECK-NEXT:  ('r_type', 0x00000115)
; CHECK-NEXT:  ('r_addend', 0x0000000000000000)
; CHECK-NEXT: ),
