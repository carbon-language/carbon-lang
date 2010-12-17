; RUN: opt < %s -simplify-libcalls -S > %t
; RUN: grep nocapture %t | count 2
; RUN: grep null %t | grep nocapture | count 1
; RUN: grep null %t | grep call | not grep readonly

; Test that we add nocapture to the declaration, and to the second call only.

declare float @strtol(i8* %s, i8** %endptr, i32 %base)

define void @foo(i8* %x, i8** %endptr) {
  call float @strtol(i8* %x, i8** %endptr, i32 10)
  call float @strtol(i8* %x, i8** null, i32 10)
  ret void
}
