; RUN: llvm-as %s -o /dev/null

; Should assemble without error.

declare void @similar_param_ptrty_callee(i8*)
define void @similar_param_ptrty(i32*) {
  musttail call void @similar_param_ptrty_callee(i8* null)
  ret void
}

declare i8* @similar_ret_ptrty_callee()
define i32* @similar_ret_ptrty() {
  %v = musttail call i8* @similar_ret_ptrty_callee()
  %w = bitcast i8* %v to i32*
  ret i32* %w
}
