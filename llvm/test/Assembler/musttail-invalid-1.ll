; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Check the error message on using ", ..." when we can't actually forward
; varargs.

%struct.A = type { i32 }

declare i8* @f(i8*, ...)

define i8* @f_thunk(i8* %this) {
  %rv = musttail call i8* (i8*, ...)* @f(i8* %this, ...)
; CHECK: error: unexpected ellipsis in argument list for musttail call in non-varargs function
  ret i8* %rv
}
