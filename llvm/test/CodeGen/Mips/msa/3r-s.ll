; Test the MSA intrinsics that are encoded with the 3R instruction format.
; There are lots of these so this covers those beginning with 's'

; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

@llvm_mips_sld_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_sld_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_sld_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_sld_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_sld_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_sld_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.sld.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_sld_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.sld.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_sld_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: sld.b
; CHECK: st.b
; CHECK: .size llvm_mips_sld_b_test
;
@llvm_mips_sld_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_sld_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_sld_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_sld_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_sld_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_sld_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.sld.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_sld_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.sld.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_sld_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: sld.h
; CHECK: st.h
; CHECK: .size llvm_mips_sld_h_test
;
@llvm_mips_sld_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_sld_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_sld_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_sld_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_sld_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_sld_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.sld.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_sld_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.sld.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_sld_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: sld.w
; CHECK: st.w
; CHECK: .size llvm_mips_sld_w_test
;
@llvm_mips_sld_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_sld_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_sld_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_sld_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_sld_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_sld_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.sld.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_sld_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.sld.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_sld_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: sld.d
; CHECK: st.d
; CHECK: .size llvm_mips_sld_d_test
;
@llvm_mips_sll_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_sll_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_sll_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_sll_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_sll_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_sll_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.sll.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_sll_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.sll.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_sll_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: sll.b
; CHECK: st.b
; CHECK: .size llvm_mips_sll_b_test
;
@llvm_mips_sll_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_sll_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_sll_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_sll_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_sll_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_sll_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.sll.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_sll_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.sll.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_sll_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: sll.h
; CHECK: st.h
; CHECK: .size llvm_mips_sll_h_test
;
@llvm_mips_sll_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_sll_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_sll_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_sll_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_sll_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_sll_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.sll.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_sll_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.sll.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_sll_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: sll.w
; CHECK: st.w
; CHECK: .size llvm_mips_sll_w_test
;
@llvm_mips_sll_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_sll_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_sll_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_sll_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_sll_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_sll_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.sll.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_sll_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.sll.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_sll_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: sll.d
; CHECK: st.d
; CHECK: .size llvm_mips_sll_d_test
;
@llvm_mips_sra_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_sra_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_sra_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_sra_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_sra_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_sra_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.sra.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_sra_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.sra.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_sra_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: sra.b
; CHECK: st.b
; CHECK: .size llvm_mips_sra_b_test
;
@llvm_mips_sra_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_sra_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_sra_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_sra_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_sra_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_sra_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.sra.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_sra_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.sra.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_sra_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: sra.h
; CHECK: st.h
; CHECK: .size llvm_mips_sra_h_test
;
@llvm_mips_sra_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_sra_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_sra_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_sra_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_sra_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_sra_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.sra.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_sra_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.sra.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_sra_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: sra.w
; CHECK: st.w
; CHECK: .size llvm_mips_sra_w_test
;
@llvm_mips_sra_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_sra_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_sra_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_sra_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_sra_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_sra_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.sra.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_sra_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.sra.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_sra_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: sra.d
; CHECK: st.d
; CHECK: .size llvm_mips_sra_d_test
;
@llvm_mips_srar_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_srar_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_srar_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_srar_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_srar_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_srar_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.srar.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_srar_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.srar.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_srar_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: srar.b
; CHECK: st.b
; CHECK: .size llvm_mips_srar_b_test
;
@llvm_mips_srar_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_srar_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_srar_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_srar_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_srar_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_srar_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.srar.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_srar_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.srar.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_srar_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: srar.h
; CHECK: st.h
; CHECK: .size llvm_mips_srar_h_test
;
@llvm_mips_srar_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_srar_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_srar_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_srar_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_srar_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_srar_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.srar.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_srar_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.srar.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_srar_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: srar.w
; CHECK: st.w
; CHECK: .size llvm_mips_srar_w_test
;
@llvm_mips_srar_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_srar_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_srar_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_srar_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_srar_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_srar_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.srar.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_srar_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.srar.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_srar_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: srar.d
; CHECK: st.d
; CHECK: .size llvm_mips_srar_d_test
;
@llvm_mips_srl_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_srl_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_srl_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_srl_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_srl_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_srl_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.srl.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_srl_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.srl.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_srl_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: srl.b
; CHECK: st.b
; CHECK: .size llvm_mips_srl_b_test
;
@llvm_mips_srl_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_srl_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_srl_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_srl_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_srl_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_srl_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.srl.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_srl_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.srl.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_srl_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: srl.h
; CHECK: st.h
; CHECK: .size llvm_mips_srl_h_test
;
@llvm_mips_srl_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_srl_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_srl_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_srl_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_srl_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_srl_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.srl.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_srl_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.srl.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_srl_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: srl.w
; CHECK: st.w
; CHECK: .size llvm_mips_srl_w_test
;
@llvm_mips_srl_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_srl_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_srl_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_srl_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_srl_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_srl_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.srl.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_srl_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.srl.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_srl_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: srl.d
; CHECK: st.d
; CHECK: .size llvm_mips_srl_d_test
;
@llvm_mips_srlr_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_srlr_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_srlr_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_srlr_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_srlr_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_srlr_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.srlr.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_srlr_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.srlr.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_srlr_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: srlr.b
; CHECK: st.b
; CHECK: .size llvm_mips_srlr_b_test
;
@llvm_mips_srlr_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_srlr_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_srlr_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_srlr_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_srlr_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_srlr_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.srlr.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_srlr_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.srlr.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_srlr_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: srlr.h
; CHECK: st.h
; CHECK: .size llvm_mips_srlr_h_test
;
@llvm_mips_srlr_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_srlr_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_srlr_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_srlr_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_srlr_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_srlr_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.srlr.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_srlr_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.srlr.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_srlr_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: srlr.w
; CHECK: st.w
; CHECK: .size llvm_mips_srlr_w_test
;
@llvm_mips_srlr_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_srlr_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_srlr_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_srlr_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_srlr_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_srlr_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.srlr.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_srlr_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.srlr.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_srlr_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: srlr.d
; CHECK: st.d
; CHECK: .size llvm_mips_srlr_d_test
;
@llvm_mips_subs_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_subs_s_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_subs_s_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_subs_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_subs_s_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_subs_s_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.subs.s.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_subs_s_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.subs.s.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_subs_s_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: subs_s.b
; CHECK: st.b
; CHECK: .size llvm_mips_subs_s_b_test
;
@llvm_mips_subs_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_subs_s_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_subs_s_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_subs_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_subs_s_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_subs_s_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.subs.s.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_subs_s_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.subs.s.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_subs_s_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: subs_s.h
; CHECK: st.h
; CHECK: .size llvm_mips_subs_s_h_test
;
@llvm_mips_subs_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_subs_s_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_subs_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_subs_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_subs_s_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_subs_s_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.subs.s.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_subs_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.subs.s.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_subs_s_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: subs_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_subs_s_w_test
;
@llvm_mips_subs_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_subs_s_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_subs_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_subs_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_subs_s_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_subs_s_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.subs.s.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_subs_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.subs.s.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_subs_s_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: subs_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_subs_s_d_test
;
@llvm_mips_subs_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_subs_u_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_subs_u_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_subs_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_subs_u_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_subs_u_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.subs.u.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_subs_u_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.subs.u.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_subs_u_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: subs_u.b
; CHECK: st.b
; CHECK: .size llvm_mips_subs_u_b_test
;
@llvm_mips_subs_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_subs_u_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_subs_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_subs_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_subs_u_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_subs_u_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.subs.u.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_subs_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.subs.u.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_subs_u_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: subs_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_subs_u_h_test
;
@llvm_mips_subs_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_subs_u_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_subs_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_subs_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_subs_u_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_subs_u_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.subs.u.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_subs_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.subs.u.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_subs_u_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: subs_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_subs_u_w_test
;
@llvm_mips_subs_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_subs_u_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_subs_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_subs_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_subs_u_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_subs_u_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.subs.u.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_subs_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.subs.u.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_subs_u_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: subs_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_subs_u_d_test
;
@llvm_mips_subsus_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_subsus_u_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_subsus_u_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_subsus_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_subsus_u_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_subsus_u_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.subsus.u.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_subsus_u_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.subsus.u.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_subsus_u_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: subsus_u.b
; CHECK: st.b
; CHECK: .size llvm_mips_subsus_u_b_test
;
@llvm_mips_subsus_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_subsus_u_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_subsus_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_subsus_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_subsus_u_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_subsus_u_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.subsus.u.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_subsus_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.subsus.u.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_subsus_u_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: subsus_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_subsus_u_h_test
;
@llvm_mips_subsus_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_subsus_u_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_subsus_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_subsus_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_subsus_u_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_subsus_u_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.subsus.u.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_subsus_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.subsus.u.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_subsus_u_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: subsus_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_subsus_u_w_test
;
@llvm_mips_subsus_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_subsus_u_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_subsus_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_subsus_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_subsus_u_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_subsus_u_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.subsus.u.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_subsus_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.subsus.u.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_subsus_u_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: subsus_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_subsus_u_d_test
;
@llvm_mips_subsuu_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_subsuu_s_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_subsuu_s_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_subsuu_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_subsuu_s_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_subsuu_s_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.subsuu.s.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_subsuu_s_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.subsuu.s.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_subsuu_s_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: subsuu_s.b
; CHECK: st.b
; CHECK: .size llvm_mips_subsuu_s_b_test
;
@llvm_mips_subsuu_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_subsuu_s_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_subsuu_s_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_subsuu_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_subsuu_s_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_subsuu_s_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.subsuu.s.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_subsuu_s_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.subsuu.s.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_subsuu_s_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: subsuu_s.h
; CHECK: st.h
; CHECK: .size llvm_mips_subsuu_s_h_test
;
@llvm_mips_subsuu_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_subsuu_s_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_subsuu_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_subsuu_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_subsuu_s_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_subsuu_s_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.subsuu.s.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_subsuu_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.subsuu.s.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_subsuu_s_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: subsuu_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_subsuu_s_w_test
;
@llvm_mips_subsuu_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_subsuu_s_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_subsuu_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_subsuu_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_subsuu_s_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_subsuu_s_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.subsuu.s.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_subsuu_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.subsuu.s.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_subsuu_s_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: subsuu_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_subsuu_s_d_test
;
@llvm_mips_subv_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_subv_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_subv_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_subv_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_subv_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_subv_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.subv.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_subv_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.subv.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_subv_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: subv.b
; CHECK: st.b
; CHECK: .size llvm_mips_subv_b_test
;
@llvm_mips_subv_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_subv_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_subv_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_subv_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_subv_h_ARG1
  %1 = load <8 x i16>* @llvm_mips_subv_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.subv.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_subv_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.subv.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_subv_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: subv.h
; CHECK: st.h
; CHECK: .size llvm_mips_subv_h_test
;
@llvm_mips_subv_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_subv_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_subv_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_subv_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_subv_w_ARG1
  %1 = load <4 x i32>* @llvm_mips_subv_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.subv.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_subv_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.subv.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_subv_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: subv.w
; CHECK: st.w
; CHECK: .size llvm_mips_subv_w_test
;
@llvm_mips_subv_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_subv_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_subv_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_subv_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_subv_d_ARG1
  %1 = load <2 x i64>* @llvm_mips_subv_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.subv.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_subv_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.subv.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_subv_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: subv.d
; CHECK: st.d
; CHECK: .size llvm_mips_subv_d_test
;
