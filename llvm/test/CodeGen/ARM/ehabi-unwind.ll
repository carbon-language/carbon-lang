; Test that the EHABI unwind instruction generator does not encounter any
; unfamiliar instructions.
; RUN: llc < %s -mtriple=thumbv7 -arm-enable-ehabi=full -disable-fp-elim
; RUN: llc < %s -mtriple=thumbv7 -arm-enable-ehabi=full
; RUN: llc < %s -mtriple=thumbv7 -arm-enable-ehabi=unwind

define void @_Z1fv() nounwind {
entry:
  ret void
}

define void @_Z1gv() nounwind {
entry:
  call void @_Z1fv()
  ret void
}
