; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -filetype=obj | llvm-readobj -r | FileCheck %s

; External symbols are a different concept to global variables but should still
; get relocations and so on when used.

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1)

define i32 @check_extern() {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* undef, i32 undef, i32 4, i1 0)
  ret i32 0
}

; CHECK: Relocations [
; CHECK:   Section (2) .rela.text {
; CHECK:     0x{{[0-9,A-F]+}} R_AARCH64_CALL26 memcpy
; CHECK:   }
; CHECK: ]
