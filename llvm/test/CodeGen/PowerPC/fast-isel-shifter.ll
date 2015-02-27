; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=ELF64

define i32 @shl() nounwind ssp {
entry:
; ELF64: shl
; ELF64: slw
  %shl = shl i32 -1, 2
  ret i32 %shl
}

define i32 @shl_reg(i32 %src1, i32 %src2) nounwind ssp {
entry:
; ELF64: shl_reg
; ELF64: slw
  %shl = shl i32 %src1, %src2
  ret i32 %shl
}

define i32 @lshr() nounwind ssp {
entry:
; ELF64: lshr
; ELF64: srw
  %lshr = lshr i32 -1, 2
  ret i32 %lshr
}

define i32 @lshr_reg(i32 %src1, i32 %src2) nounwind ssp {
entry:
; ELF64: lshr_reg
; ELF64: srw
  %lshr = lshr i32 %src1, %src2
  ret i32 %lshr
}

define i32 @ashr() nounwind ssp {
entry:
; ELF64: ashr
; ELF64: srawi
  %ashr = ashr i32 -1, 2
  ret i32 %ashr
}

define i32 @ashr_reg(i32 %src1, i32 %src2) nounwind ssp {
entry:
; ELF64: ashr_reg
; ELF64: sraw
  %ashr = ashr i32 %src1, %src2
  ret i32 %ashr
}

