; RUN: llc -march aarch64 %s -o - | FileCheck %s

@g0 = external global <3 x float>, align 16
@g1 = external global <3 x float>, align 4

; CHECK: ldr s[[R0:[0-9]+]], {{\[}}[[R1:x[0-9]+]]{{\]}}, #4
; CHECK: ld1{{\.?s?}} { v[[R0]]{{\.?s?}} }[1], {{\[}}[[R1]]{{\]}}
; CHECK: str d[[R0]]

define void @blam() {
  %tmp4 = getelementptr inbounds <3 x float>, <3 x float>* @g1, i64 0, i64 0
  %tmp5 = load <3 x float>, <3 x float>* @g0, align 16
  %tmp6 = extractelement <3 x float> %tmp5, i64 0
  store float %tmp6, float* %tmp4
  %tmp7 = getelementptr inbounds float, float* %tmp4, i64 1
  %tmp8 = load <3 x float>, <3 x float>* @g0, align 16
  %tmp9 = extractelement <3 x float> %tmp8, i64 1
  store float %tmp9, float* %tmp7
  ret void;
}
