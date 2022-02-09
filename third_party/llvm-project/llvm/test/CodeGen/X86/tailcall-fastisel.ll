; RUN: llc < %s -mtriple=x86_64-apple-darwin -tailcallopt -fast-isel -fast-isel-abort=1 | FileCheck %s

%0 = type { i64, i32, i8* }

define fastcc i8* @"visit_array_aux<`Reference>"(%0 %arg, i32 %arg1) nounwind {
fail:                                             ; preds = %entry
  %tmp20 = tail call fastcc i8* @"visit_array_aux<`Reference>"(%0 %arg, i32 undef) ; <i8*> [#uses=1]
; CHECK: jmp "_visit_array_aux<`Reference>" ## TAILCALL
  ret i8* %tmp20
}

define i32 @foo() nounwind {
entry:
 %0 = tail call i32 (...) @bar() nounwind       ; <i32> [#uses=1]
 ret i32 %0
}

declare i32 @bar(...) nounwind
