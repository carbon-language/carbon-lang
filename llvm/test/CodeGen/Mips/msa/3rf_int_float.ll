; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

@llvm_mips_fceq_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fceq_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fceq_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fceq_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fceq_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fceq_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fceq.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fceq_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fceq.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fceq_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fceq.w
; CHECK: st.w
; CHECK: .size llvm_mips_fceq_w_test
;
@llvm_mips_fceq_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fceq_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fceq_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fceq_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fceq_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fceq_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fceq.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fceq_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fceq.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fceq_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fceq.d
; CHECK: st.d
; CHECK: .size llvm_mips_fceq_d_test
;
@llvm_mips_fcge_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcge_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcge_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcge_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fcge_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fcge_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcge.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fcge_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcge.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcge_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcge.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcge_w_test
;
@llvm_mips_fcge_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcge_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcge_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcge_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fcge_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fcge_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcge.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fcge_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcge.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcge_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcge.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcge_d_test
;
@llvm_mips_fcgt_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcgt_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcgt_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcgt_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fcgt_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fcgt_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcgt.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fcgt_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcgt.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcgt_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcgt.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcgt_w_test
;
@llvm_mips_fcgt_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcgt_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcgt_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcgt_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fcgt_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fcgt_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcgt.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fcgt_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcgt.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcgt_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcgt.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcgt_d_test
;
@llvm_mips_fcle_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcle_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcle_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcle_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fcle_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fcle_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcle.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fcle_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcle.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcle_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcle.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcle_w_test
;
@llvm_mips_fcle_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcle_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcle_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcle_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fcle_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fcle_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcle.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fcle_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcle.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcle_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcle.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcle_d_test
;
@llvm_mips_fclt_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fclt_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fclt_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fclt_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fclt_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fclt_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fclt.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fclt_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fclt.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fclt_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fclt.w
; CHECK: st.w
; CHECK: .size llvm_mips_fclt_w_test
;
@llvm_mips_fclt_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fclt_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fclt_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fclt_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fclt_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fclt_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fclt.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fclt_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fclt.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fclt_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fclt.d
; CHECK: st.d
; CHECK: .size llvm_mips_fclt_d_test
;
@llvm_mips_fcne_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcne_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcne_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcne_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fcne_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fcne_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcne.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fcne_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcne.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcne_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcne.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcne_w_test
;
@llvm_mips_fcne_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcne_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcne_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcne_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fcne_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fcne_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcne.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fcne_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcne.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcne_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcne.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcne_d_test
;
@llvm_mips_fcun_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcun_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcun_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcun_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fcun_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fcun_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcun.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fcun_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcun.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcun_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcun.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcun_w_test
;
@llvm_mips_fcun_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcun_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcun_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcun_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fcun_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fcun_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcun.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fcun_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcun.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcun_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcun.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcun_d_test
;
@llvm_mips_fseq_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fseq_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fseq_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fseq_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fseq_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fseq_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fseq.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fseq_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fseq.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fseq_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fseq.w
; CHECK: st.w
; CHECK: .size llvm_mips_fseq_w_test
;
@llvm_mips_fseq_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fseq_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fseq_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fseq_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fseq_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fseq_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fseq.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fseq_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fseq.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fseq_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fseq.d
; CHECK: st.d
; CHECK: .size llvm_mips_fseq_d_test
;
@llvm_mips_fsge_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsge_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsge_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsge_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fsge_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fsge_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsge.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fsge_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsge.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsge_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsge.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsge_w_test
;
@llvm_mips_fsge_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsge_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsge_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsge_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fsge_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fsge_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsge.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fsge_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsge.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsge_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsge.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsge_d_test
;
@llvm_mips_fsgt_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsgt_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsgt_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsgt_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fsgt_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fsgt_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsgt.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fsgt_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsgt.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsgt_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsgt.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsgt_w_test
;
@llvm_mips_fsgt_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsgt_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsgt_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsgt_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fsgt_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fsgt_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsgt.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fsgt_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsgt.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsgt_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsgt.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsgt_d_test
;
@llvm_mips_fsle_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsle_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsle_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsle_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fsle_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fsle_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsle.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fsle_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsle.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsle_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsle.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsle_w_test
;
@llvm_mips_fsle_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsle_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsle_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsle_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fsle_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fsle_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsle.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fsle_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsle.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsle_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsle.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsle_d_test
;
@llvm_mips_fslt_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fslt_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fslt_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fslt_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fslt_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fslt_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fslt.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fslt_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fslt.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fslt_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fslt.w
; CHECK: st.w
; CHECK: .size llvm_mips_fslt_w_test
;
@llvm_mips_fslt_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fslt_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fslt_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fslt_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fslt_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fslt_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fslt.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fslt_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fslt.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fslt_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fslt.d
; CHECK: st.d
; CHECK: .size llvm_mips_fslt_d_test
;
@llvm_mips_fsne_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsne_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsne_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsne_w_test() nounwind {
entry:
  %0 = load <4 x float>* @llvm_mips_fsne_w_ARG1
  %1 = load <4 x float>* @llvm_mips_fsne_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsne.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_fsne_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsne.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsne_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsne.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsne_w_test
;
@llvm_mips_fsne_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsne_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsne_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsne_d_test() nounwind {
entry:
  %0 = load <2 x double>* @llvm_mips_fsne_d_ARG1
  %1 = load <2 x double>* @llvm_mips_fsne_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsne.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_fsne_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsne.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsne_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsne.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsne_d_test
;
