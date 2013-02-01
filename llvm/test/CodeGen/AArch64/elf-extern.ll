; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -filetype=obj | elf-dump | FileCheck %s

; External symbols are a different concept to global variables but should still
; get relocations and so on when used.

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1)

define i32 @check_extern() {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* undef, i32 undef, i32 4, i1 0)
  ret i32 0
}

; CHECK: .rela.text
; CHECK: ('r_sym', 0x00000009)
; CHECK-NEXT: ('r_type', 0x0000011b)

; CHECK: .symtab
; CHECK: Symbol 9
; CHECK-NEXT: memcpy


