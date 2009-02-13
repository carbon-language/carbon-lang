; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis > %t
; RUN: grep nocapture %t | count 2
; RUN: grep null %t | grep nocapture | count 1

; Test that we do add nocapture to the declaration, and to the second call only.

declare float @strtof(i8* %s, i8** %endptr, i32 %base)

define void @foo(i8* %x, i8** %endptr) {
  call float @strtof(i8* %x, i8** %endptr, i32 0)
  call float @strtof(i8* %x, i8** null, i32 0)
  ret void
}
