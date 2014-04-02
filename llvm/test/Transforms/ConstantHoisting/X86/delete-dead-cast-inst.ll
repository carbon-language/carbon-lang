; Test if this compiles without assertions.
; RUN: opt -S -consthoist < %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%T = type { i32, i32, i32, i32 }

define i32 @test1() nounwind {
  %base = inttoptr i32 12345678 to %T*
  %addr1 = getelementptr %T* %base, i32 0, i32 1
  %addr2 = getelementptr %T* %base, i32 0, i32 2
  %addr3 = getelementptr %T* %base, i32 0, i32 3
  ret i32 12345678
}

