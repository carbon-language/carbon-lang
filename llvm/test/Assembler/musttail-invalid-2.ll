; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Check the error message on skipping ", ..." at the end of a musttail call argument list.

%struct.A = type { i32 }

declare i8* @f(i8*, ...)

define i8* @f_thunk(i8* %this, ...) {
  %rv = musttail call i8* (i8*, ...)* @f(i8* %this)
; CHECK: error: expected '...' at end of argument list for musttail call in varargs function
  ret i8* %rv
}
