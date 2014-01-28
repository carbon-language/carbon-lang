; RUN: llc < %s -mtriple thumbv7-linux -filetype=obj -o %t
; Two pass decoding needed because llvm-objdump does not respect mapping symbols
; RUN: llvm-objdump -triple thumbv7-linux -d %t | FileCheck --check-prefix=THUMB %s
; RUN: llvm-objdump -triple armv7-linux   -d %t | FileCheck --check-prefix=ARM   %s

; THUMB: foo:
; THUMB:  a:       18 47                                           bx      r3
; THUMB: 28:       70 47                                           bx      lr

; ARM: foo:
; ARM:   10:       04 70 2d e5                                     str     r7, [sp, #-4]!

define void @foo(i8* %start, i64 %size) {
entry:
  %0 = ptrtoint i8* %start to i32
  %conv = zext i32 %0 to i64
  %add = add i64 %conv, %size
  %conv1 = trunc i64 %add to i32
  %1 = tail call i32 asm sideeffect "@   Enter ARM Mode  \0A\09adr r3, 1f      \0A\09bx  r3          \0A\09.align 4        \0A\09.arm            \0A1:  push {r7}       \0A\09mov r7, $4      \0A\09add r1, r2, r3  \0A\09pop {r7}        \0A\09@   Enter THUMB Mode\0A\09adr r3, 2f+1    \0A\09bx  r3          \0A\09.thumb          \0A2:                  \0A\09", "={r0},{r0},{r1},{r2},r,~{r3}"(i32 %0, i32 %conv1, i32 0, i32 254)
  ret void
}


