; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

; Test that we add nocapture to the declaration, and to the second call only.

; CHECK: declare float @strtol(i8*, i8** nocapture, i32) nounwind
declare float @strtol(i8* %s, i8** %endptr, i32 %base)

define void @foo(i8* %x, i8** %endptr) {
; CHECK:  call float @strtol(i8* %x, i8** %endptr, i32 10)
  call float @strtol(i8* %x, i8** %endptr, i32 10)
; CHECK: %2 = call float @strtol(i8* nocapture %x, i8** null, i32 10)
  call float @strtol(i8* %x, i8** null, i32 10)
  ret void
}
