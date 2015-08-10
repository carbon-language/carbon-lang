; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

target triple = "nvptx-unknown-cuda"

declare void @bar(<4 x i32>)

; CHECK-LABEL: @foo
define void @foo(<4 x i32> %a) {
; CHECK: st.param.v4.b32
  tail call void @bar(<4 x i32> %a)
  ret void
}
