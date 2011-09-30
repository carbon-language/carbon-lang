; RUN: llc -march=mips64el -mcpu=mips64r1 < %s | FileCheck %s

define i64 @f0(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: dsllv
  %shl = shl i64 %a0, %a1
  ret i64 %shl
}

define i64 @f1(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: dsrav
  %shr = ashr i64 %a0, %a1
  ret i64 %shr
}

define i64 @f2(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: dsrlv
  %shr = lshr i64 %a0, %a1
  ret i64 %shr
}

define i64 @f3(i64 %a0) nounwind readnone {
entry:
; CHECK: dsll
  %shl = shl i64 %a0, 10
  ret i64 %shl
}

define i64 @f4(i64 %a0) nounwind readnone {
entry:
; CHECK: dsra
  %shr = ashr i64 %a0, 10
  ret i64 %shr
}

define i64 @f5(i64 %a0) nounwind readnone {
entry:
; CHECK: dsrl
  %shr = lshr i64 %a0, 10
  ret i64 %shr
}

define i64 @f6(i64 %a0) nounwind readnone {
entry:
; CHECK: dsll32
  %shl = shl i64 %a0, 40
  ret i64 %shl
}

define i64 @f7(i64 %a0) nounwind readnone {
entry:
; CHECK: dsra32
  %shr = ashr i64 %a0, 40
  ret i64 %shr
}

define i64 @f8(i64 %a0) nounwind readnone {
entry:
; CHECK: dsrl32
  %shr = lshr i64 %a0, 40
  ret i64 %shr
}
