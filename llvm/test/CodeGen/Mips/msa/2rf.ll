; Test the MSA intrinsics that are encoded with the 2RF instruction format.

; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

@llvm_mips_flog2_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_flog2_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_flog2_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_flog2_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.flog2.w(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_flog2_w_RES
  ret void
}

declare <4 x float> @llvm.mips.flog2.w(<4 x float>) nounwind

; CHECK: llvm_mips_flog2_w_test:
; CHECK: ld.w
; CHECK: flog2.w
; CHECK: st.w
; CHECK: .size llvm_mips_flog2_w_test
;
@llvm_mips_flog2_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_flog2_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_flog2_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_flog2_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.flog2.d(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_flog2_d_RES
  ret void
}

declare <2 x double> @llvm.mips.flog2.d(<2 x double>) nounwind

; CHECK: llvm_mips_flog2_d_test:
; CHECK: ld.d
; CHECK: flog2.d
; CHECK: st.d
; CHECK: .size llvm_mips_flog2_d_test
;
@llvm_mips_frint_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_frint_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_frint_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_frint_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.frint.w(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_frint_w_RES
  ret void
}

declare <4 x float> @llvm.mips.frint.w(<4 x float>) nounwind

; CHECK: llvm_mips_frint_w_test:
; CHECK: ld.w
; CHECK: frint.w
; CHECK: st.w
; CHECK: .size llvm_mips_frint_w_test
;
@llvm_mips_frint_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_frint_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_frint_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_frint_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.frint.d(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_frint_d_RES
  ret void
}

declare <2 x double> @llvm.mips.frint.d(<2 x double>) nounwind

; CHECK: llvm_mips_frint_d_test:
; CHECK: ld.d
; CHECK: frint.d
; CHECK: st.d
; CHECK: .size llvm_mips_frint_d_test
;
@llvm_mips_frcp_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_frcp_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_frcp_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_frcp_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.frcp.w(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_frcp_w_RES
  ret void
}

declare <4 x float> @llvm.mips.frcp.w(<4 x float>) nounwind

; CHECK: llvm_mips_frcp_w_test:
; CHECK: ld.w
; CHECK: frcp.w
; CHECK: st.w
; CHECK: .size llvm_mips_frcp_w_test
;
@llvm_mips_frcp_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_frcp_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_frcp_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_frcp_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.frcp.d(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_frcp_d_RES
  ret void
}

declare <2 x double> @llvm.mips.frcp.d(<2 x double>) nounwind

; CHECK: llvm_mips_frcp_d_test:
; CHECK: ld.d
; CHECK: frcp.d
; CHECK: st.d
; CHECK: .size llvm_mips_frcp_d_test
;
@llvm_mips_frsqrt_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_frsqrt_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_frsqrt_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_frsqrt_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.frsqrt.w(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_frsqrt_w_RES
  ret void
}

declare <4 x float> @llvm.mips.frsqrt.w(<4 x float>) nounwind

; CHECK: llvm_mips_frsqrt_w_test:
; CHECK: ld.w
; CHECK: frsqrt.w
; CHECK: st.w
; CHECK: .size llvm_mips_frsqrt_w_test
;
@llvm_mips_frsqrt_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_frsqrt_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_frsqrt_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_frsqrt_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.frsqrt.d(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_frsqrt_d_RES
  ret void
}

declare <2 x double> @llvm.mips.frsqrt.d(<2 x double>) nounwind

; CHECK: llvm_mips_frsqrt_d_test:
; CHECK: ld.d
; CHECK: frsqrt.d
; CHECK: st.d
; CHECK: .size llvm_mips_frsqrt_d_test
;
@llvm_mips_fsqrt_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsqrt_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fsqrt_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fsqrt_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.fsqrt.w(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_fsqrt_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fsqrt.w(<4 x float>) nounwind

; CHECK: llvm_mips_fsqrt_w_test:
; CHECK: ld.w
; CHECK: fsqrt.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsqrt_w_test
;
@llvm_mips_fsqrt_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsqrt_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fsqrt_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fsqrt_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.fsqrt.d(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_fsqrt_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fsqrt.d(<2 x double>) nounwind

; CHECK: llvm_mips_fsqrt_d_test:
; CHECK: ld.d
; CHECK: fsqrt.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsqrt_d_test
;
