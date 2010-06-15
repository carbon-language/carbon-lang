; RUN: llc < %s -march=arm
; PR4716

define void @_start() nounwind naked {
entry:
  tail call  void @exit(i32 undef) noreturn nounwind
  unreachable
}

declare void @exit(i32) noreturn nounwind
