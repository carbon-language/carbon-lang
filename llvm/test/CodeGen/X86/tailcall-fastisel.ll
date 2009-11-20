; RUN: llc < %s -march=x86-64 -tailcallopt -fast-isel | grep TAILCALL

; Fast-isel shouldn't attempt to handle this tail call, and it should
; cleanly terminate instruction selection in the block after it's
; done to avoid emitting invalid MachineInstrs.

%0 = type { i64, i32, i8* }

define fastcc i8* @"visit_array_aux<`Reference>"(%0 %arg, i32 %arg1) nounwind {
fail:                                             ; preds = %entry
  %tmp20 = tail call fastcc i8* @"visit_array_aux<`Reference>"(%0 %arg, i32 undef) ; <i8*> [#uses=1]
  ret i8* %tmp20
}
