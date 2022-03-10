; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; CHECK-NOT: .align 2
define ptx_device void @foo() align 2 {
; CHECK-LABEL: .func foo
  ret void
}
