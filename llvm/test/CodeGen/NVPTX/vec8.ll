; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

target triple = "nvptx-unknown-cuda"

; CHECK: .visible .func foo
define void @foo(<8 x i8> %a, i8* %b) {
  %t0 = extractelement <8 x i8> %a, i32 0
; CHECK-DAG: ld.param.v4.u8
; CHECK-DAG: ld.param.u32
  store i8 %t0, i8* %b
  ret void
}

