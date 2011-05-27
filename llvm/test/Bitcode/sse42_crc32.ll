; Check to make sure old CRC32 intrinsics are auto-upgraded
; correctly.
;
; Rdar: 9472944
;
; RUN: llvm-dis < %s.bc | not grep {i32 @llvm.x86.sse42.crc32.8(}
; RUN: llvm-dis < %s.bc | grep {i32 @llvm.x86.sse42.crc32.32.8(}
; RUN: llvm-dis < %s.bc | not grep {i32 @llvm.x86.sse42.crc32.16(}
; RUN: llvm-dis < %s.bc | grep {i32 @llvm.x86.sse42.crc32.32.16(}
; RUN: llvm-dis < %s.bc | not grep {i32 @llvm.x86.sse42.crc32.32(}
; RUN: llvm-dis < %s.bc | grep {i32 @llvm.x86.sse42.crc32.32.32(}
; RUN: llvm-dis < %s.bc | not grep {i64 @llvm.x86.sse42.crc64.8(}
; RUN: llvm-dis < %s.bc | grep {i64 @llvm.x86.sse42.crc32.64.8(}
; RUN: llvm-dis < %s.bc | not grep {i64 @llvm.x86.sse42.crc64.8(}
; RUN: llvm-dis < %s.bc | grep {i64 @llvm.x86.sse42.crc32.64.8(}

