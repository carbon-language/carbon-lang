; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin9 -mattr=sse42 | FileCheck %s -check-prefix=X32
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin9 -mattr=sse42 | FileCheck %s -check-prefix=X64

declare i32 @llvm.x86.sse42.crc32.8(i32, i8) nounwind
declare i32 @llvm.x86.sse42.crc32.16(i32, i16) nounwind
declare i32 @llvm.x86.sse42.crc32.32(i32, i32) nounwind

define i32 @crc32_8(i32 %a, i8 %b) nounwind {
  %tmp = call i32 @llvm.x86.sse42.crc32.8(i32 %a, i8 %b)
  ret i32 %tmp
; X32: _crc32_8:
; X32:     crc32   8(%esp), %eax

; X64: _crc32_8:
; X64:     crc32   %sil, %eax
}


define i32 @crc32_16(i32 %a, i16 %b) nounwind {
  %tmp = call i32 @llvm.x86.sse42.crc32.16(i32 %a, i16 %b)
  ret i32 %tmp
; X32: _crc32_16:
; X32:     crc32   8(%esp), %eax

; X64: _crc32_16:
; X64:     crc32   %si, %eax
}


define i32 @crc32_32(i32 %a, i32 %b) nounwind {
  %tmp = call i32 @llvm.x86.sse42.crc32.32(i32 %a, i32 %b)
  ret i32 %tmp
; X32: _crc32_32:
; X32:     crc32   8(%esp), %eax

; X64: _crc32_32:
; X64:     crc32   %esi, %eax
}
