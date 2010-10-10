; Test that, for a 64 bit signed div, a libcall to alldiv is made on Windows
; except for cygwin.

; RUN: llc < %s -mtriple i386-pc-win32 | FileCheck %s
; RUN: llc < %s -mtriple i386-pc-cygwin | FileCheck %s -check-prefix CYGWIN

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind readonly {
entry:
  %conv4 = sext i32 %argc to i64
  %div = sdiv i64 84, %conv4
  %conv7 = trunc i64 %div to i32
  ret i32 %conv7
}

; CHECK: alldiv
; CYGWIN: divdi3
