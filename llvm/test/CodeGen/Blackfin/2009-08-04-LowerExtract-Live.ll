; RUN: llc < %s -march=bfin -join-liveintervals=0 -verify-machineinstrs

; Provoke an error in LowerSubregsPass::LowerExtract where the live range of a
; super-register is illegally extended.

define i16 @f(i16 %x1, i16 %x2, i16 %x3, i16 %x4) {
  %y1 = add i16 %x1, 1
  %y2 = add i16 %x2, 2
  %y3 = add i16 %x3, 3
  %y4 = add i16 %x4, 4
  %z12 = add i16 %y1, %y2
  %z34 = add i16 %y3, %y4
  %p = add i16 %z12, %z34
  ret i16 %p
}
