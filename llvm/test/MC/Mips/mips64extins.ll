; RUN: llc -march=mips64el -filetype=obj -mcpu=mips64r2 -mattr=n64 %s -o - \
; RUN: | llvm-objdump -disassemble -triple mips64el -mattr +mips64r2 - \
; RUN: | FileCheck %s

define i64 @dext(i64 %i) nounwind readnone {
entry:
; CHECK: dext ${{[0-9]+}}, ${{[0-9]+}}, 5, 10
  %shr = lshr i64 %i, 5
  %and = and i64 %shr, 1023
  ret i64 %and
}

define i64 @dextu(i64 %i) nounwind readnone {
entry:
; CHECK: dextu ${{[0-9]+}}, ${{[0-9]+}}, 2, 6
  %shr = lshr i64 %i, 34
  %and = and i64 %shr, 63
  ret i64 %and
}

define i64 @dextm(i64 %i) nounwind readnone {
entry:
; CHECK: dextm ${{[0-9]+}}, ${{[0-9]+}}, 5, 2
  %shr = lshr i64 %i, 5
  %and = and i64 %shr, 17179869183
  ret i64 %and
}

