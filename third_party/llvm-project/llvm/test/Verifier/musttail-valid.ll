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

declare x86_thiscallcc void @varargs_thiscall(i8*, ...)
define x86_thiscallcc void @varargs_thiscall_thunk(i8* %this, ...) {
  musttail call x86_thiscallcc void (i8*, ...) @varargs_thiscall(i8* %this, ...)
  ret void
}

declare x86_fastcallcc void @varargs_fastcall(i8*, ...)
define x86_fastcallcc void @varargs_fastcall_thunk(i8* %this, ...) {
  musttail call x86_fastcallcc void (i8*, ...) @varargs_fastcall(i8* %this, ...)
  ret void
}

define x86_thiscallcc void @varargs_thiscall_unreachable(i8* %this, ...) {
  unreachable
}

define x86_thiscallcc void @varargs_thiscall_ret_unreachable(i8* %this, ...) {
  musttail call x86_thiscallcc void (i8*, ...) @varargs_thiscall(i8* %this, ...)
  ret void
bb1:
  ret void
}
