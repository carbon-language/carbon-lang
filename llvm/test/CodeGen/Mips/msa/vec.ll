; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s
; 
; Test the MSA intrinsics that are encoded with the VEC instruction format.

@llvm_mips_and_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_and_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_and_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_and_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_and_v_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_and_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.and.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_and_v_b_RES
  ret void
}

; CHECK: llvm_mips_and_v_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: and.v
; CHECK: st.b
; CHECK: .size llvm_mips_and_v_b_test
;
@llvm_mips_bmnz_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bmnz_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_bmnz_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bmnz_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_bmnz_v_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_bmnz_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.bmnz.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_bmnz_v_b_RES
  ret void
}

; CHECK: llvm_mips_bmnz_v_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: bmnz.v
; CHECK: st.b
; CHECK: .size llvm_mips_bmnz_v_b_test
;
@llvm_mips_bmz_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bmz_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_bmz_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bmz_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_bmz_v_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_bmz_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.bmz.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_bmz_v_b_RES
  ret void
}

; CHECK: llvm_mips_bmz_v_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: bmz.v
; CHECK: st.b
; CHECK: .size llvm_mips_bmz_v_b_test
;
@llvm_mips_bmz_v_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bmz_v_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_bmz_v_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

@llvm_mips_bsel_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bsel_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_bsel_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bsel_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_bsel_v_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_bsel_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.bsel.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_bsel_v_b_RES
  ret void
}

; CHECK: llvm_mips_bsel_v_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: bsel.v
; CHECK: st.b
; CHECK: .size llvm_mips_bsel_v_b_test
;
@llvm_mips_nor_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_nor_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_nor_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_nor_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_nor_v_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_nor_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.nor.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_nor_v_b_RES
  ret void
}

; CHECK: llvm_mips_nor_v_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: nor.v
; CHECK: st.b
; CHECK: .size llvm_mips_nor_v_b_test
;
@llvm_mips_or_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_or_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_or_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_or_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_or_v_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_or_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.or.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_or_v_b_RES
  ret void
}

; CHECK: llvm_mips_or_v_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: or.v
; CHECK: st.b
; CHECK: .size llvm_mips_or_v_b_test
;
@llvm_mips_xor_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_xor_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_xor_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_xor_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_xor_v_b_ARG1
  %1 = load <16 x i8>* @llvm_mips_xor_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.xor.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_xor_v_b_RES
  ret void
}

; CHECK: llvm_mips_xor_v_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: xor.v
; CHECK: st.b
; CHECK: .size llvm_mips_xor_v_b_test
;
declare <16 x i8> @llvm.mips.and.v(<16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.bmnz.v(<16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.bmz.v(<16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.bsel.v(<16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.nor.v(<16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.or.v(<16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.xor.v(<16 x i8>, <16 x i8>) nounwind
