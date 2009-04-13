; RUN: llvm-as < %s | llc -march=x86-64 > %t
; RUN: grep mov %t | count 6
; RUN: grep {movb	%ah, (%rsi)} %t | count 3
; RUN: llvm-as < %s | llc -march=x86 > %t
; RUN: grep mov %t | count 3
; RUN: grep {movb	%ah, (%e} %t | count 3

; Use h-register extract and store.

define void @foo16(i16 inreg %p, i8* inreg %z) nounwind {
  %q = lshr i16 %p, 8
  %t = trunc i16 %q to i8
  store i8 %t, i8* %z
  ret void
}
define void @foo32(i32 inreg %p, i8* inreg %z) nounwind {
  %q = lshr i32 %p, 8
  %t = trunc i32 %q to i8
  store i8 %t, i8* %z
  ret void
}
define void @foo64(i64 inreg %p, i8* inreg %z) nounwind {
  %q = lshr i64 %p, 8
  %t = trunc i64 %q to i8
  store i8 %t, i8* %z
  ret void
}
