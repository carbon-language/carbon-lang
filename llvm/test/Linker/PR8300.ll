; RUN: echo {%foo2 = type \{ \[8 x i8\] \} \
; RUN:       declare void @zed(%foo2*) } > %t.ll
; RUN: llvm-link %t.ll %s -o %t.bc

%foo = type { [8 x i8] }
%bar = type { [9 x i8] }

@zed = alias bitcast (void (%bar*)* @xyz to void (%foo*)*)

define void @xyz(%bar* %this) {
entry:
  ret void
}
