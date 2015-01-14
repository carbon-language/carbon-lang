; RUN: llc -O1 < %s -march=mips64 -mcpu=octeon | FileCheck %s -check-prefix=OCTEON
; RUN: llc -O1 < %s -march=mips64 -mcpu=mips64 | FileCheck %s -check-prefix=MIPS64

define i64 @addi64(i64 %a, i64 %b) nounwind {
entry:
; OCTEON-LABEL: addi64:
; OCTEON: jr      $ra
; OCTEON: baddu   $2, $4, $5
; MIPS64-LABEL: addi64:
; MIPS64: daddu
; MIPS64: jr
; MIPS64: andi
  %add = add i64 %a, %b
  %and = and i64 %add, 255
  ret i64 %and
}

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

define i64 @cmpeq(i64 %a, i64 %b) nounwind {
entry:
; OCTEON-LABEL: cmpeq:
; OCTEON: jr     $ra
; OCTEON: seq    $2, $4, $5
; MIPS64-LABEL: cmpeq:
; MIPS64: xor    $1, $4, $5
; MIPS64: sltiu  $1, $1, 1
; MIPS64: dsll   $1, $1, 32
; MIPS64: jr     $ra
; MIPS64: dsrl   $2, $1, 32
  %res = icmp eq i64 %a, %b
  %res2 = zext i1 %res to i64
  ret i64 %res2
}

define i64 @cmpeqi(i64 %a) nounwind {
entry:
; OCTEON-LABEL: cmpeqi:
; OCTEON: jr     $ra
; OCTEON: seqi   $2, $4, 42
; MIPS64-LABEL: cmpeqi:
; MIPS64: daddiu $1, $zero, 42
; MIPS64: xor    $1, $4, $1
; MIPS64: sltiu  $1, $1, 1
; MIPS64: dsll   $1, $1, 32
; MIPS64: jr     $ra
; MIPS64: dsrl   $2, $1, 32
  %res = icmp eq i64 %a, 42
  %res2 = zext i1 %res to i64
  ret i64 %res2
}

define i64 @cmpne(i64 %a, i64 %b) nounwind {
entry:
; OCTEON-LABEL: cmpne:
; OCTEON: jr     $ra
; OCTEON: sne    $2, $4, $5
; MIPS64-LABEL: cmpne:
; MIPS64: xor    $1, $4, $5
; MIPS64: sltu   $1, $zero, $1
; MIPS64: dsll   $1, $1, 32
; MIPS64: jr     $ra
; MIPS64: dsrl   $2, $1, 32
  %res = icmp ne i64 %a, %b
  %res2 = zext i1 %res to i64
  ret i64 %res2
}

define i64 @cmpnei(i64 %a) nounwind {
entry:
; OCTEON-LABEL: cmpnei:
; OCTEON: jr     $ra
; OCTEON: snei   $2, $4, 42
; MIPS64-LABEL: cmpnei:
; MIPS64: daddiu $1, $zero, 42
; MIPS64: xor    $1, $4, $1
; MIPS64: sltu   $1, $zero, $1
; MIPS64: dsll   $1, $1, 32
; MIPS64: jr     $ra
; MIPS64: dsrl   $2, $1, 32
  %res = icmp ne i64 %a, 42
  %res2 = zext i1 %res to i64
  ret i64 %res2
}
