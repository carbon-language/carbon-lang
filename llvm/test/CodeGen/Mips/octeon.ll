; RUN: llc -O1 < %s -march=mips64 -mcpu=octeon | FileCheck %s -check-prefix=OCTEON
; RUN: llc -O1 < %s -march=mips64 -mcpu=mips64 | FileCheck %s -check-prefix=MIPS64

define i64 @mul(i64 %a, i64 %b) nounwind {
entry:
; OCTEON-LABEL: mul:
; OCTEON: jr    $ra
; OCTEON: dmul  $2, $4, $5
; MIPS64-LABEL: mul:
; MIPS64: dmult
; MIPS64: jr
; MIPS64: mflo
  %res = mul i64 %a, %b
  ret i64 %res
}
