; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=32
; RUN: llc -march=mipsel -mattr=+fp64 < %s | FileCheck %s -check-prefix=32-FP64
; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s -check-prefix=64

; 32:      addiu  $sp, $sp, -8
; 32-FP64: addiu  $sp, $sp, -16
; 64:      addiu  $sp, $sp, -16

define i32 @foo1() #0 {
entry:
  ret i32 14
}

attributes #0 = { "no-frame-pointer-elim"="true" }
