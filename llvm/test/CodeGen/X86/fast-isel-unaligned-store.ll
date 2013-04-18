; RUN: llc -mtriple=x86_64-none-linux -fast-isel -fast-isel-abort < %s | FileCheck %s
; RUN: llc -mtriple=i686-none-linux -fast-isel -fast-isel-abort < %s | FileCheck %s

define i32 @test_store_32(i32* nocapture %addr, i32 %value) {
entry:
  store i32 %value, i32* %addr, align 1
  ret i32 %value
}

; CHECK: ret

define i16 @test_store_16(i16* nocapture %addr, i16 %value) {
entry:
  store i16 %value, i16* %addr, align 1
  ret i16 %value
}

; CHECK: ret
