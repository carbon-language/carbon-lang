; RUN: llc < %s
; PR4975

%0 = type <{ [0 x i32] }>
%union.T0 = type { }

@.str = private constant [1 x i8] c" "

define arm_apcscc void @t(%0) nounwind {
entry:
  %arg0 = alloca %union.T0
  %1 = bitcast %union.T0* %arg0 to %0*
  store %0 %0, %0* %1, align 1
  ret void
}

declare arm_apcscc i32 @printf(i8*, ...)
