; Test the MSA intrinsics that are encoded with the 2R instruction format.

; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

@llvm_mips_nloc_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_nloc_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_nloc_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_nloc_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.nloc.b(<16 x i8> %0)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_nloc_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.nloc.b(<16 x i8>) nounwind

; CHECK: llvm_mips_nloc_b_test:
; CHECK: ld.b
; CHECK: nloc.b
; CHECK: st.b
; CHECK: .size llvm_mips_nloc_b_test
;
@llvm_mips_nloc_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_nloc_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_nloc_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_nloc_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.nloc.h(<8 x i16> %0)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_nloc_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.nloc.h(<8 x i16>) nounwind

; CHECK: llvm_mips_nloc_h_test:
; CHECK: ld.h
; CHECK: nloc.h
; CHECK: st.h
; CHECK: .size llvm_mips_nloc_h_test
;
@llvm_mips_nloc_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_nloc_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_nloc_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_nloc_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.nloc.w(<4 x i32> %0)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_nloc_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.nloc.w(<4 x i32>) nounwind

; CHECK: llvm_mips_nloc_w_test:
; CHECK: ld.w
; CHECK: nloc.w
; CHECK: st.w
; CHECK: .size llvm_mips_nloc_w_test
;
@llvm_mips_nloc_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_nloc_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_nloc_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_nloc_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.nloc.d(<2 x i64> %0)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_nloc_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.nloc.d(<2 x i64>) nounwind

; CHECK: llvm_mips_nloc_d_test:
; CHECK: ld.d
; CHECK: nloc.d
; CHECK: st.d
; CHECK: .size llvm_mips_nloc_d_test
;
@llvm_mips_nlzc_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_nlzc_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_nlzc_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_nlzc_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.nlzc.b(<16 x i8> %0)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_nlzc_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.nlzc.b(<16 x i8>) nounwind

; CHECK: llvm_mips_nlzc_b_test:
; CHECK: ld.b
; CHECK: nlzc.b
; CHECK: st.b
; CHECK: .size llvm_mips_nlzc_b_test
;
@llvm_mips_nlzc_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_nlzc_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_nlzc_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_nlzc_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.nlzc.h(<8 x i16> %0)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_nlzc_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.nlzc.h(<8 x i16>) nounwind

; CHECK: llvm_mips_nlzc_h_test:
; CHECK: ld.h
; CHECK: nlzc.h
; CHECK: st.h
; CHECK: .size llvm_mips_nlzc_h_test
;
@llvm_mips_nlzc_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_nlzc_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_nlzc_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_nlzc_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.nlzc.w(<4 x i32> %0)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_nlzc_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.nlzc.w(<4 x i32>) nounwind

; CHECK: llvm_mips_nlzc_w_test:
; CHECK: ld.w
; CHECK: nlzc.w
; CHECK: st.w
; CHECK: .size llvm_mips_nlzc_w_test
;
@llvm_mips_nlzc_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_nlzc_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_nlzc_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_nlzc_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.nlzc.d(<2 x i64> %0)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_nlzc_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.nlzc.d(<2 x i64>) nounwind

; CHECK: llvm_mips_nlzc_d_test:
; CHECK: ld.d
; CHECK: nlzc.d
; CHECK: st.d
; CHECK: .size llvm_mips_nlzc_d_test
;
@llvm_mips_pcnt_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_pcnt_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_pcnt_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_pcnt_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.pcnt.b(<16 x i8> %0)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_pcnt_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.pcnt.b(<16 x i8>) nounwind

; CHECK: llvm_mips_pcnt_b_test:
; CHECK: ld.b
; CHECK: pcnt.b
; CHECK: st.b
; CHECK: .size llvm_mips_pcnt_b_test
;
@llvm_mips_pcnt_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_pcnt_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_pcnt_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_pcnt_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.pcnt.h(<8 x i16> %0)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_pcnt_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.pcnt.h(<8 x i16>) nounwind

; CHECK: llvm_mips_pcnt_h_test:
; CHECK: ld.h
; CHECK: pcnt.h
; CHECK: st.h
; CHECK: .size llvm_mips_pcnt_h_test
;
@llvm_mips_pcnt_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_pcnt_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_pcnt_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_pcnt_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.pcnt.w(<4 x i32> %0)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_pcnt_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.pcnt.w(<4 x i32>) nounwind

; CHECK: llvm_mips_pcnt_w_test:
; CHECK: ld.w
; CHECK: pcnt.w
; CHECK: st.w
; CHECK: .size llvm_mips_pcnt_w_test
;
@llvm_mips_pcnt_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_pcnt_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_pcnt_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_pcnt_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.pcnt.d(<2 x i64> %0)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_pcnt_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.pcnt.d(<2 x i64>) nounwind

; CHECK: llvm_mips_pcnt_d_test:
; CHECK: ld.d
; CHECK: pcnt.d
; CHECK: st.d
; CHECK: .size llvm_mips_pcnt_d_test
;
