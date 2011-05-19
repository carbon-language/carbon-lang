; RUN: llc < %s -march=x86-64 -mattr=sse42 | FileCheck %s

; crc32 with 64-bit destination zeros high 32-bit.
; rdar://9467055

define i64 @t() nounwind {
entry:
; CHECK: t:
; CHECK: crc32q
; CHECK-NOT: mov
; CHECK-NEXT: crc32q
  %0 = tail call i64 @llvm.x86.sse42.crc64.64(i64 0, i64 4) nounwind
  %1 = and i64 %0, 4294967295
  %2 = tail call i64 @llvm.x86.sse42.crc64.64(i64 %1, i64 4) nounwind
  %3 = and i64 %2, 4294967295
  ret i64 %3
}

declare i64 @llvm.x86.sse42.crc64.64(i64, i64) nounwind readnone
