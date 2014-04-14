; RUN: llc -march=mips64el -mcpu=mips4 < %s | FileCheck -check-prefix=CHECK -check-prefix=MIPS4 %s
; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck -check-prefix=CHECK -check-prefix=MIPS64 %s

define i64 @t1(i64 %X) nounwind readnone {
entry:
; CHECK-LABEL: t1:
; MIPS4-NOT: dclz
; MIPS64: dclz
  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %X, i1 true)
  ret i64 %tmp1
}

declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone

define i64 @t3(i64 %X) nounwind readnone {
entry:
; CHECK-LABEL: t3:
; MIPS4-NOT: dclo
; MIPS64: dclo
  %neg = xor i64 %X, -1
  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %neg, i1 true)
  ret i64 %tmp1
}

