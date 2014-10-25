; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 | FileCheck %s

declare <2 x float> @bar(<2 x float> %input)

define void @foo(<2 x float> %input, <2 x float>* %output) {
; CHECK-LABEL: @foo
entry:
  %call = tail call <2 x float> @bar(<2 x float> %input)
; CHECK: .param .align 8 .b8 retval0[8];
; CHECK: ld.param.v2.f32 {[[ELEM1:%f[0-9]+]], [[ELEM2:%f[0-9]+]]}, [retval0+0];
  store <2 x float> %call, <2 x float>* %output, align 8
; CHECK: st.v2.f32 [{{%rd[0-9]+}}], {[[ELEM1]], [[ELEM2]]}
  ret void
}
