; Check that (or (shl x, y), (srl x, (sub 32, y))) is folded into (rotl x, y)
; and (or (shl x, (sub 32, y)), (srl x, r)) into (rotr x, y) even if the
; argument is zero extended. Fix for PR16726.

; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s

define zeroext i8 @rolbyte(i32 %nBits_arg, i8 %x_arg) nounwind readnone {
entry:
  %tmp1 = zext i8 %x_arg to i32
  %tmp3 = shl i32 %tmp1, %nBits_arg
  %tmp8 = sub i32 8, %nBits_arg
  %tmp10 = lshr i32 %tmp1, %tmp8
  %tmp11 = or i32 %tmp3, %tmp10
  %tmp12 = trunc i32 %tmp11 to i8
  ret i8 %tmp12
}
; CHECK:    rolb %cl, %{{[a-z0-9]+}}


define zeroext i8 @rorbyte(i32 %nBits_arg, i8 %x_arg) nounwind readnone {
entry:
  %tmp1 = zext i8 %x_arg to i32
  %tmp3 = lshr i32 %tmp1, %nBits_arg
  %tmp8 = sub i32 8, %nBits_arg
  %tmp10 = shl i32 %tmp1, %tmp8
  %tmp11 = or i32 %tmp3, %tmp10
  %tmp12 = trunc i32 %tmp11 to i8
  ret i8 %tmp12
}
; CHECK:    rorb %cl, %{{[a-z0-9]+}}

define zeroext i16 @rolword(i32 %nBits_arg, i16 %x_arg) nounwind readnone {
entry:
  %tmp1 = zext i16 %x_arg to i32
  %tmp3 = shl i32 %tmp1, %nBits_arg
  %tmp8 = sub i32 16, %nBits_arg
  %tmp10 = lshr i32 %tmp1, %tmp8
  %tmp11 = or i32 %tmp3, %tmp10
  %tmp12 = trunc i32 %tmp11 to i16
  ret i16 %tmp12
}
; CHECK:    rolw %cl, %{{[a-z0-9]+}}

define zeroext i16 @rorword(i32 %nBits_arg, i16 %x_arg) nounwind readnone {
entry:
  %tmp1 = zext i16 %x_arg to i32
  %tmp3 = lshr i32 %tmp1, %nBits_arg
  %tmp8 = sub i32 16, %nBits_arg
  %tmp10 = shl i32 %tmp1, %tmp8
  %tmp11 = or i32 %tmp3, %tmp10
  %tmp12 = trunc i32 %tmp11 to i16
  ret i16 %tmp12
}
; CHECK:    rorw %cl, %{{[a-z0-9]+}}

define i64 @roldword(i64 %nBits_arg, i32 %x_arg) nounwind readnone {
entry:
  %tmp1 = zext i32 %x_arg to i64
  %tmp3 = shl i64 %tmp1, %nBits_arg
  %tmp8 = sub i64 32, %nBits_arg
  %tmp10 = lshr i64 %tmp1, %tmp8
  %tmp11 = or i64 %tmp3, %tmp10
  ret i64 %tmp11
}
; CHECK:    roll %cl, %{{[a-z0-9]+}}

define zeroext i64 @rordword(i64 %nBits_arg, i32 %x_arg) nounwind readnone {
entry:
  %tmp1 = zext i32 %x_arg to i64
  %tmp3 = lshr i64 %tmp1, %nBits_arg
  %tmp8 = sub i64 32, %nBits_arg
  %tmp10 = shl i64 %tmp1, %tmp8
  %tmp11 = or i64 %tmp3, %tmp10
  ret i64 %tmp11
}
; CHECK:    rorl %cl, %{{[a-z0-9]+}}
