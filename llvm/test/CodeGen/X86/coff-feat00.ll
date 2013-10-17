; RUN: llc -O0 -mtriple=i386-pc-win32 -filetype=asm -o - %s | FileCheck %s --check-prefix=WIN32
; RUN: llc -O0 -mtriple=i386-pc-cygwin -filetype=asm -o - %s | FileCheck %s --check-prefix=CYGWIN

define i32 @foo() {
  ret i32 0
}

; WIN32: "@feat.00" = 1
; CYGWIN-NOT: "@feat.00" = 1
