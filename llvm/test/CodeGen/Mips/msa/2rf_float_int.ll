; Test the MSA integer to floating point conversion intrinsics that are encoded
; with the 2RF instruction format.

; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

@llvm_mips_ffint_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ffint_s_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_ffint_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_ffint_s_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.ffint.s.w(<4 x i32> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_ffint_s_w_RES
  ret void
}

declare <4 x float> @llvm.mips.ffint.s.w(<4 x i32>) nounwind

; CHECK: llvm_mips_ffint_s_w_test:
; CHECK: ld.w
; CHECK: ffint_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_ffint_s_w_test
;
@llvm_mips_ffint_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_ffint_s_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_ffint_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_ffint_s_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.ffint.s.d(<2 x i64> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_ffint_s_d_RES
  ret void
}

declare <2 x double> @llvm.mips.ffint.s.d(<2 x i64>) nounwind

; CHECK: llvm_mips_ffint_s_d_test:
; CHECK: ld.d
; CHECK: ffint_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_ffint_s_d_test
;
@llvm_mips_ffint_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ffint_u_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_ffint_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_ffint_u_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.ffint.u.w(<4 x i32> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_ffint_u_w_RES
  ret void
}

declare <4 x float> @llvm.mips.ffint.u.w(<4 x i32>) nounwind

; CHECK: llvm_mips_ffint_u_w_test:
; CHECK: ld.w
; CHECK: ffint_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_ffint_u_w_test
;
@llvm_mips_ffint_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_ffint_u_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_ffint_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_ffint_u_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.ffint.u.d(<2 x i64> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_ffint_u_d_RES
  ret void
}

declare <2 x double> @llvm.mips.ffint.u.d(<2 x i64>) nounwind

; CHECK: llvm_mips_ffint_u_d_test:
; CHECK: ld.d
; CHECK: ffint_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_ffint_u_d_test
;
