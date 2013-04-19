; RUN: llc < %s -mtriple=i386-unknown-mingw32 | FileCheck %s

; Check that a fastcall function gets correct mangling

define x86_fastcallcc void @func(i64 %X, i8 %Y, i8 %G, i16 %Z) {
; CHECK: @func@20:
        ret void
}

define x86_fastcallcc i32 @"\01DoNotMangle"(i32 %a) {
; CHECK: DoNotMangle:
entry:
  ret i32 %a
}
