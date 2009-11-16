; RUN: llc < %s | FileCheck %s
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-n32"
target triple = "mips-unknown-linux"

define float @h() nounwind readnone {
entry:
; CHECK: lui $2, %hi($CPI1_0)
; CHECK: lwc1 $f0, %lo($CPI1_0)($2)
  ret float 0x400B333340000000
}
