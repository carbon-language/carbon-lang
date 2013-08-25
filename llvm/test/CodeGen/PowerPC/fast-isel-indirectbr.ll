; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=ELF64

define void @t1(i8* %x) {
entry:
; ELF64: t1
  br label %L0

L0:
  br label %L1

L1:
  indirectbr i8* %x, [ label %L0, label %L1 ]
; ELF64: mtctr 3
; ELF64: bctr
}
