; RUN: llc  < %s -march=mipsel -mcpu=mips32r2 | FileCheck %s -check-prefix=MIPS32
; RUN: llc  < %s -march=mips64el -mcpu=mips64r2 | FileCheck %s -check-prefix=MIPS64

declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1)

define <2 x i32> @ctlzv2i32(<2 x i32> %x) {
entry:
; MIPS32: clz     $2, $4
; MIPS32: jr      $ra
; MIPS32: clz     $3, $5

; MIPS64: clz     $2, $4
; MIPS64: jr      $ra
; MIPS64: clz     $3, $5

  %ret = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %x, i1 true)
  ret <2 x i32> %ret
}

