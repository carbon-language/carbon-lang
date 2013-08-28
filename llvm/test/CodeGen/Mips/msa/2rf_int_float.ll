; Test the MSA floating point to integer intrinsics that are encoded with the
; 2RF instruction format. This includes conversions but other instructions such
; as fclass are also here.

; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

@llvm_mips_fclass_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fclass_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fclass_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fclass_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.fclass.w(<4 x float> %0)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_fclass_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fclass.w(<4 x float>) nounwind

; CHECK: llvm_mips_fclass_w_test:
; CHECK: ld.w
; CHECK: fclass.w
; CHECK: st.w
; CHECK: .size llvm_mips_fclass_w_test
;
@llvm_mips_fclass_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fclass_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fclass_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fclass_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.fclass.d(<2 x double> %0)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_fclass_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fclass.d(<2 x double>) nounwind

; CHECK: llvm_mips_fclass_d_test:
; CHECK: ld.d
; CHECK: fclass.d
; CHECK: st.d
; CHECK: .size llvm_mips_fclass_d_test
;
@llvm_mips_ftrunc_s_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_ftrunc_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ftrunc_s_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_ftrunc_s_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.ftrunc.s.w(<4 x float> %0)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_ftrunc_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ftrunc.s.w(<4 x float>) nounwind

; CHECK: llvm_mips_ftrunc_s_w_test:
; CHECK: ld.w
; CHECK: ftrunc_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_ftrunc_s_w_test
;
@llvm_mips_ftrunc_s_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_ftrunc_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ftrunc_s_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_ftrunc_s_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.ftrunc.s.d(<2 x double> %0)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_ftrunc_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ftrunc.s.d(<2 x double>) nounwind

; CHECK: llvm_mips_ftrunc_s_d_test:
; CHECK: ld.d
; CHECK: ftrunc_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_ftrunc_s_d_test
;
@llvm_mips_ftrunc_u_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_ftrunc_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ftrunc_u_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_ftrunc_u_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.ftrunc.u.w(<4 x float> %0)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_ftrunc_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ftrunc.u.w(<4 x float>) nounwind

; CHECK: llvm_mips_ftrunc_u_w_test:
; CHECK: ld.w
; CHECK: ftrunc_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_ftrunc_u_w_test
;
@llvm_mips_ftrunc_u_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_ftrunc_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ftrunc_u_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_ftrunc_u_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.ftrunc.u.d(<2 x double> %0)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_ftrunc_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ftrunc.u.d(<2 x double>) nounwind

; CHECK: llvm_mips_ftrunc_u_d_test:
; CHECK: ld.d
; CHECK: ftrunc_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_ftrunc_u_d_test
;
@llvm_mips_ftint_s_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_ftint_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ftint_s_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_ftint_s_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.ftint.s.w(<4 x float> %0)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_ftint_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ftint.s.w(<4 x float>) nounwind

; CHECK: llvm_mips_ftint_s_w_test:
; CHECK: ld.w
; CHECK: ftint_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_ftint_s_w_test
;
@llvm_mips_ftint_s_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_ftint_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ftint_s_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_ftint_s_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.ftint.s.d(<2 x double> %0)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_ftint_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ftint.s.d(<2 x double>) nounwind

; CHECK: llvm_mips_ftint_s_d_test:
; CHECK: ld.d
; CHECK: ftint_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_ftint_s_d_test
;
@llvm_mips_ftint_u_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_ftint_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ftint_u_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_ftint_u_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.ftint.u.w(<4 x float> %0)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_ftint_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ftint.u.w(<4 x float>) nounwind

; CHECK: llvm_mips_ftint_u_w_test:
; CHECK: ld.w
; CHECK: ftint_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_ftint_u_w_test
;
@llvm_mips_ftint_u_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_ftint_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ftint_u_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_ftint_u_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.ftint.u.d(<2 x double> %0)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_ftint_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ftint.u.d(<2 x double>) nounwind

; CHECK: llvm_mips_ftint_u_d_test:
; CHECK: ld.d
; CHECK: ftint_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_ftint_u_d_test
;
