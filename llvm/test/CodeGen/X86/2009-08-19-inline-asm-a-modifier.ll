; RUN: llvm-as < %s | llc | FileCheck %s
; ModuleID = 'asm.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define i32 @main() nounwind {
entry:
; CHECK: %gs:6
  %asmtmp.i = tail call i16 asm "movw\09%gs:${1:a}, ${0:w}", "=r,ir,~{dirflag},~{fpsr},~{flags}"(i32 6) nounwind ; <i16> [#uses=1]
  %0 = zext i16 %asmtmp.i to i32                  ; <i32> [#uses=1]
  ret i32 %0
}

define zeroext i16 @readgsword2(i32 %address) nounwind {
entry:
; CHECK: %gs:(%eax)
  %asmtmp = tail call i16 asm "movw\09%gs:${1:a}, ${0:w}", "=r,ir,~{dirflag},~{fpsr},~{flags}"(i32 %address) nounwind ; <i16> [#uses=1]
  ret i16 %asmtmp
}
