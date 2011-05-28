; Check to make sure old CRC32 intrinsics are auto-upgraded
; correctly.
;
; Rdar: 9472944
;
; RUN: llvm-dis < %s.bc | FileCheck %s

; crc32.8 should upgrade to crc32.32.8
; CHECK: i32 @llvm.x86.sse42.crc32.32.8(
; CHECK-NOT: i32 @llvm.x86.sse42.crc32.8(

; crc32.16 should upgrade to crc32.32.16
; CHECK: i32 @llvm.x86.sse42.crc32.32.16(
; CHECK-NOT: i32 @llvm.x86.sse42.crc32.16(

; crc32.32 should upgrade to crc32.32.32
; CHECK: i32 @llvm.x86.sse42.crc32.32.32(
; CHECK-NOT: i32 @llvm.x86.sse42.crc32.32(

; crc64.8 should upgrade to crc32.64.8
; CHECK: i64 @llvm.x86.sse42.crc32.64.8(
; CHECK-NOT: i64 @llvm.x86.sse42.crc64.8(

; crc64.64 should upgrade to crc32.64.64
; CHECK: i64 @llvm.x86.sse42.crc32.64.64(
; CHECK-NOT: i64 @llvm.x86.sse42.crc64.64(


