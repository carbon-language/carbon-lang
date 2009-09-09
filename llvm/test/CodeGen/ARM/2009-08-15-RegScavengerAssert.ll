; RUN: llc < %s -march=arm
; PR4716

define arm_aapcscc void @_start() nounwind naked {
entry:
  tail call arm_aapcscc  void @exit(i32 undef) noreturn nounwind
  unreachable
}

declare arm_aapcscc void @exit(i32) noreturn nounwind
