; RUN: llvm-as < %s | llc -march=x86 > %t
; RUN: not grep movz %t
; RUN: not grep cmov %t
; RUN: grep movb %t | count 2

; Don't try to use a 16-bit conditional move to do an 8-bit select,
; because it isn't worth it. Just use a branch instead.

define i8 @foo(i1 inreg %c, i8 inreg %a, i8 inreg %b) {
  %d = select i1 %c, i8 %a, i8 %b
  ret i8 %d
}
