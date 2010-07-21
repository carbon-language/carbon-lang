; RUN: llc < %s -mtriple=arm-linux-gnueabi -mattr=+vfp2 | FileCheck %s -check-prefix=ELF
; RUN: llc < %s -mtriple=arm-apple-darwin -mattr=+vfp2 | FileCheck %s -check-prefix=DARWIN

define i32 @f1(i32 %a, i64 %b) {
; ELF: f1:
; ELF: mov r0, r2
; DARWIN: f1:
; DARWIN: mov r0, r1
        %tmp = call i32 @g1(i64 %b)
        ret i32 %tmp
}

; test that allocating the double to r2/r3 makes r1 unavailable on gnueabi.
define i32 @f2() nounwind optsize {
; ELF: f2:
; ELF: mov  r0, #128
; ELF: str  r0, [sp]
; DARWIN: f2:
; DARWIN: mov	r3, #128
entry:
  %0 = tail call i32 (i32, ...)* @g2(i32 5, double 1.600000e+01, i32 128) nounwind optsize ; <i32> [#uses=1]
  %not. = icmp ne i32 %0, 128                     ; <i1> [#uses=1]
  %.0 = zext i1 %not. to i32                      ; <i32> [#uses=1]
  ret i32 %.0
}

declare i32 @g1(i64)

declare i32 @g2(i32 %i, ...)
