; Check to make sure old CRC32 intrinsics are auto-upgraded
; correctly.
;
; Rdar: 9472944
;
; RUN: opt < %s | llvm-dis | FileCheck %s

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


define void @foo() nounwind readnone ssp {
entry:
  %0 = call i32 @llvm.x86.sse42.crc32.8(i32 0, i8 0)
  %1 = call i32 @llvm.x86.sse42.crc32.16(i32 0, i16 0)
  %2 = call i32 @llvm.x86.sse42.crc32.32(i32 0, i32 0)
  %3 = call i64 @llvm.x86.sse42.crc64.8(i64 0, i8 0)
  %4 = call i64 @llvm.x86.sse42.crc64.64(i64 0, i64 0)
  ret void
}

declare i32 @llvm.x86.sse42.crc32.8(i32, i8) nounwind readnone
declare i32 @llvm.x86.sse42.crc32.16(i32, i16) nounwind readnone
declare i32 @llvm.x86.sse42.crc32.32(i32, i32) nounwind readnone
declare i64 @llvm.x86.sse42.crc64.8(i64, i8) nounwind readnone
declare i64 @llvm.x86.sse42.crc64.64(i64, i64) nounwind readnone
